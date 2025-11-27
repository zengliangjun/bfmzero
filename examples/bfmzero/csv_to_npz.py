"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python examples/bfmzero/csv_to_npz.py --input_file /workspace.data1/ISAACSIM45/MATA/data/SPLITDATA/csv/g1/LAFAN1_Retargeting_Dataset/dance1_subject1.csv --input_fps 30 \
    --output_name /workspace.data1/ISAACSIM45/MATA/data/SPLITDATA/motions/g1/LAFAN1_Retargeting_Dataset/dance1_subject1.npz --output_fps 50 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=False, help="The path to the input motion csv file.")
parser.add_argument("--out_dir", type=str, required=False, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=False, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

#args_cli.input_file = "/workspace/data2/VSCODE/RL_MOTION_TRACKING/MOTIONS_DATASETS/OMOMO/GMR/g1/csv/sub3_woodchair_026.csv"
#args_cli.input_fps = 50
#args_cli.output_name = "OMOMO/sub3_woodchair_026"
#args_cli.headless = True


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import sys
import os.path as osp
root0 = osp.join(osp.dirname(__file__), "../..")
if args_cli.out_dir is not None and osp.exists(args_cli.out_dir):
    root0 = args_cli.out_dir

if args_cli.output_name is None:
    args_cli.output_name = osp.basename(args_cli.input_file).split(".")[0]

source_dir = f"{root0}"
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils

##
# Pre-defined configs
##
from metamotivo.fb_bfmzero.isaac.robots.g1 import G1_CYLINDER_CFG
from metamotivo.fb_bfmzero.isaac import isaac_utils

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    contact_sensors = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor], bool
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag





def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot: Articulation = scene["robot"]
    contact_sensors = scene["contact_sensors"]



    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    feet_indexes = robot.find_bodies(".*_ankle_roll_link", preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:

            policy_items = {
                        "base_ang_vel" : robot.data.root_ang_vel_b.cpu().numpy(),
                        "gravity" : robot.data.projected_gravity_b.cpu().numpy(),
                        #"joint_pos" : robot.data.joint_pos - robot.data.default_joint_pos,
                        #"joint_vel" : robot.data.joint_vel - robot.data.default_joint_vel,
                    }
            


            lin_vel_w_ref_b = robot.data.body_lin_vel_w
            quat_w = torch.repeat_interleave(robot.data.root_quat_w[:, None, :], lin_vel_w_ref_b.shape[1], dim=1)
            local_body_lin_vel = math_utils.quat_apply_inverse(quat_w, lin_vel_w_ref_b).reshape(quat_w.shape[0], -1)

            body_ang_vel_w = robot.data.body_ang_vel_w
            local_body_ang_vel = math_utils.quat_apply_inverse(quat_w, body_ang_vel_w).reshape(quat_w.shape[0], -1)

            root_pos_w = robot.data.root_pos_w[:, None, :]
            body_pos_w = robot.data.body_pos_w
            local_pos_w = body_pos_w - root_pos_w
            local_body_pos = math_utils.quat_apply_inverse(quat_w, local_pos_w).reshape(quat_w.shape[0], -1)

            body_quat_w = robot.data.body_quat_w
            inv_quat_w = math_utils.quat_inv(robot.data.root_quat_w)
            inv_quat_w = torch.repeat_interleave(inv_quat_w[:, None, :], body_quat_w.shape[1], dim=1)

            lb_quat = math_utils.quat_mul(inv_quat_w, body_quat_w)
            lb_quat = math_utils.normalize(lb_quat)

            #lb_mat = math_utils.matrix_from_quat(lb_quat)[..., :2]
            #local_body_quat = lb_mat.reshape(lb_mat.shape[0], -1)
            local_body_quat = isaac_utils.quat_to_tan_norm_wxyz(lb_quat)

            # check if contact force is above threshold
            net_contact_forces = contact_sensors.data.net_forces_w_history  # N, T, B, 3
            forces = torch.max(torch.norm(net_contact_forces[:, :, feet_indexes], dim=-1), dim=1)[0]
            feet_contact = forces > 1
            feet_status = feet_contact.float()

            net_contact_forces = contact_sensors.data.net_forces_w[:, feet_indexes]
            feet_forces = net_contact_forces.flatten(1)

            critic_items = {
                        "base_lin_vel": robot.data.root_lin_vel_b.cpu().numpy(),
                        "base_pos_z": robot.data.root_pos_w[:, 2:].cpu().numpy(),

                        "local_body_lin_vel": local_body_lin_vel.cpu().numpy(),
                        "local_body_ang_vel": local_body_ang_vel.cpu().numpy(),
                        "local_body_pos": local_body_pos.cpu().numpy(),
                        "local_body_quat": local_body_quat.cpu().numpy(),

                        "joint_acc": robot.data.joint_acc.cpu().numpy(),
                        "joint_stiffness": (robot.data.joint_stiffness / robot.data.default_joint_stiffness).cpu().numpy(),
                        "joint_damping": (robot.data.joint_damping / robot.data.default_joint_damping).cpu().numpy(),
                        "friction_coeff": robot.data.joint_friction_coeff.cpu().numpy(),
                        "torques": robot.data.applied_torque.cpu().numpy(),

                        "feet_status": feet_status.cpu().numpy(),
                        "feet_forces": feet_forces.cpu().numpy()
                    }

            pos_items = {
                "joint_pos": robot.data.joint_pos.cpu().numpy(),
                "joint_vel": robot.data.joint_vel.cpu().numpy(),
                "body_pos_w": robot.data.body_pos_w.cpu().numpy(),
                "body_quat_w": robot.data.body_quat_w.cpu().numpy(),
                "body_lin_vel_w": robot.data.body_lin_vel_w.cpu().numpy(),
                "body_ang_vel_w": robot.data.body_ang_vel_w.cpu().numpy(),
            }

            items = {}
            items.update(policy_items)
            items.update(critic_items)
            items.update(pos_items)


            for key, value in items.items():
                if key in log:
                    log[key].append(value)
                else:
                    log[key] = [value]

        if reset_flag and not file_saved:
            file_saved = True
            for key, value in log.items():
                log[key] = np.concatenate(value, axis=0)

            # np.savez("/tmp/motion.npz", **log)
            global root0
            output_root = osp.join(root0, "motions")
            path = osp.join(output_root, f"{args_cli.output_name}.npz")
            parent_path = osp.dirname(path)
            if not osp.exists(parent_path):
                import os
                os.makedirs(parent_path)
            np.savez(path, **log)

            print(f"[INFO]: Motion saved to {parent_path}")

            return

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
