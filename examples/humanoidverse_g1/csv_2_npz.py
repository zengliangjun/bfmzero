import os
import sys
import os.path as osp
root = osp.join(osp.dirname(__file__), "../..")

if root not in sys.path:
    sys.path.insert(0, root)

os.chdir(root)

from humanoidverse_env import motions_main
#motions_main.humanoidverse_start()
motions_main.humanoidverse_headless()

import torch
from humanoidverse_env.motions import csv_loader
from humanoidverse.envs.legged_base_task import legged_robot_base

humanoidverse_task: legged_robot_base.LeggedRobotBase = motions_main.humanoidverse_task


def walk_files(input_dir):
    if not osp.isdir(input_dir) and input_dir.endswith(".csv"):
        return [input_dir]

    inputfiles = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            full_file = osp.join(root, file)
            inputfiles.append(full_file)
    return inputfiles

def run_motion(motion, input_joint_ids, out_file):

    log = {

    }

    while True:
        state, reset_flag = motion.get_next_state()

        ##
        dof_states = torch.zeros((humanoidverse_task.num_envs, \
                                  humanoidverse_task.num_dof, 2), \
                                  device=str(humanoidverse_task.device))

        dof_states[..., 0] = humanoidverse_task.default_dof_pos

        dof_states[..., input_joint_ids, 0] = state["dof_pos"]
        dof_states[..., input_joint_ids, 1] = state["dof_vel"]

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

        obs_dict, rew_buf, reset_buf, extras = humanoidverse_task.step_motions(step_motion)

        for key, value in step_motion.items():
            if key in log:
                log[key].append(value.to("cpu"))
            else:
                log[key] = [value.to("cpu")]

        for key, value in obs_dict.items():
            if key in log:
                log[key].append(value.to("cpu"))
            else:
                log[key] = [value.to("cpu")]

        if reset_flag:
            break

    for key, value in log.items():
        log[key] = torch.cat(value, axis=0)

    torch.save(log, out_file)
    print(f"[INFO]: Motion saved to {out_file}")

def run_simulator(input_files, input_joint_ids, base_dir, out_dir):
    """Runs the simulation loop."""
    obs_dict = humanoidverse_task.reset_all()

    for file in input_files:
        motion = csv_loader.MotionLoader(
                motion_file=file,
                input_fps=30,
                output_fps=50,
                device=humanoidverse_task.device
            )

        out_file = file.replace(base_dir, out_dir)
        out_file = out_file.replace("csv", "pth")

        out_full_dir = osp.dirname(out_file)
        if not osp.exists(out_full_dir):
            os.makedirs(out_full_dir)

        run_motion(motion, input_joint_ids, out_file)

if __name__ == "__main__":

    input_dir = "/workspace/data/csv/g1/LAFAN1_Retargeting_Dataset/"
    base_dir = "/workspace/data/csv/"
    out_dir = "/workspace/data/motions/"

    input_files = walk_files(input_dir)

    input_joint_names=[
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
            "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]

    input_joint_ids = humanoidverse_task.get_motion_joint(input_joint_names)

    output_joint_names=[
            'left_hip_pitch_joint', 'right_hip_pitch_joint',
            'waist_yaw_joint',
            'left_hip_roll_joint', 'right_hip_roll_joint',
            'waist_roll_joint',
            'left_hip_yaw_joint', 'right_hip_yaw_joint',
            'waist_pitch_joint',
            'left_knee_joint', 'right_knee_joint',
            'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
            'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
            'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
            'left_ankle_roll_joint', 'right_ankle_roll_joint',
            'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
            'left_elbow_joint', 'right_elbow_joint',
            'left_wrist_roll_joint', 'right_wrist_roll_joint',
            'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
            'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
        ]

    output_body_names = [
            'pelvis',
            'left_hip_pitch_link', 'right_hip_pitch_link',
            'waist_yaw_link',
            'left_hip_roll_link', 'right_hip_roll_link',
            'waist_roll_link',
            'left_hip_yaw_link', 'right_hip_yaw_link',
            'torso_link',
            'left_knee_link', 'right_knee_link',
            'left_shoulder_pitch_link', 'right_shoulder_pitch_link',
            'left_ankle_pitch_link', 'right_ankle_pitch_link',
            'left_shoulder_roll_link', 'right_shoulder_roll_link',
            'left_ankle_roll_link', 'right_ankle_roll_link',
            'left_shoulder_yaw_link', 'right_shoulder_yaw_link',
            'left_elbow_link', 'right_elbow_link',
            'left_wrist_roll_link', 'right_wrist_roll_link',
            'left_wrist_pitch_link', 'right_wrist_pitch_link',
            'left_wrist_yaw_link', 'right_wrist_yaw_link'
        ]

    output_joint_ids = humanoidverse_task.get_motion_joint(output_joint_names)
    output_body_ids = humanoidverse_task.get_motion_body(output_body_names)

    run_simulator(
        input_files,
        input_joint_ids,
        base_dir,
        out_dir
    )

    motions_main.humanoidverse_final()
