import os
import sys
import os.path as osp
root = osp.join(osp.dirname(__file__), "../..")

if root not in sys.path:
    sys.path.insert(0, root)

os.chdir(root)

from humanoidverse_env import motions_main
motions_main.humanoidverse_start()

import torch
from humanoidverse_env.motions import csv_loader
from humanoidverse.envs.legged_base_task import legged_robot_base

humanoidverse_task: legged_robot_base.LeggedRobotBase = motions_main.humanoidverse_task

def run_simulator(joint_names):
    """Runs the simulation loop."""
    # Load motion
    motion = csv_loader.MotionLoader(
        motion_file="/workspace/data/csv/g1/LAFAN1_Retargeting_Dataset/dance1_subject1.csv",
        input_fps=30,
        output_fps=50,
        device=humanoidverse_task.device
    )
    joint_ids = humanoidverse_task.set_motion_joint(joint_names)

    obs_dict = humanoidverse_task.reset_all()
    while True:
        state, reset_flag = motion.get_next_state()

        ##
        dof_states = torch.zeros((humanoidverse_task.num_envs, \
                                  humanoidverse_task.num_dof, 2), \
                                  device=str(humanoidverse_task.device))

        dof_states[..., 0] = humanoidverse_task.default_dof_pos

        dof_states[..., joint_ids, 0] = state["dof_pos"]
        dof_states[..., joint_ids, 1] = state["dof_vel"]

        ##
        root_states = torch.repeat_interleave(humanoidverse_task.base_init_state[None, :], \
                                              humanoidverse_task.num_envs, dim = 0)
        root_states[:, :3] = state["base_pos"]
        base_rot = state["base_rot"][:, [1, 2, 3, 0]] # wxyz -> xyzw
        root_states[:, 3:7] = base_rot

        root_states[:, 7:10] = state["base_lin_vel"]
        root_states[:, 10:] = state["base_ang_vel"]

        step_motion = {
            "dof_states": dof_states,
            "root_states": root_states
        }

        humanoidverse_task.step_motions(step_motion)


if __name__ == "__main__":
    run_simulator(
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
