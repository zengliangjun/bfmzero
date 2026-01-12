import os
import sys
import os.path as osp
root = osp.join(osp.dirname(__file__), "../..")

if root not in sys.path:
    sys.path.insert(0, root)

os.chdir(root)

from humanoidverse_env import motions_main
motions_main.humanoidverse_play()

import torch
from humanoidverse_env.motions import npz_loader
from humanoidverse.envs.legged_base_task import legged_robot_base

humanoidverse_task: legged_robot_base.LeggedRobotBase = motions_main.humanoidverse_task

def run_simulator(joint_names, body_names):
    """Runs the simulation loop."""
    # Load motion
    motion = npz_loader.MotionLoader(
        motion_file="/workspace/data2/motion/AMASS/bfmzero/g1_50fps/AMASS/HDM05/dg/HDM_dg_02-02_01_120_stageii.pth",
        device=humanoidverse_task.device
    )
    # joint_ids = humanoidverse_task.set_motion_joint(joint_names)
    # body_ids = humanoidverse_task.set_motion_body(body_names)


    obs_dict = humanoidverse_task.reset_all()
    while True:
        state, reset_flag = motion.get_next_state()

        ###
        obs_dict, rew_buf, reset_buf, extras = humanoidverse_task.step_motions(state)

        if reset_flag:
            motions_main.humanoidverse_final()
            return


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
        body_names = [
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
    )
