import os
import sys
import os.path as osp
root = osp.join(osp.dirname(__file__), "..")

if root not in sys.path:
    sys.path.insert(0, root)

os.chdir(root)

from humanoidverse.envs.motions_task import humanoidverse_main
humanoidverse_main.humanoidverse_start()

import torch
from humanoidverse_env.motions import npz_loader
from humanoidverse.envs.legged_base_task import legged_robot_base

humanoidverse_task: legged_robot_base.LeggedRobotBase = humanoidverse_main.humanoidverse_task

def run_simulator(joint_names, body_names):
    """Runs the simulation loop."""
    # Load motion
    motion = npz_loader.MotionLoader(
        motion_file="/workspace/data/motions/g1/LAFAN1_Retargeting_Dataset/dance1_subject1.npz",
        device=humanoidverse_task.device
    )
    joint_ids = humanoidverse_task.set_motion_joint(joint_names)
    body_ids = humanoidverse_task.set_motion_body(body_names)


    obs_dict = humanoidverse_task.reset_all()
    while True:
        state, reset_flag = motion.get_next_state()

        ##
        dof_states = torch.zeros((humanoidverse_task.num_envs, \
                                  humanoidverse_task.num_dof, 2), \
                                  device=str(humanoidverse_task.device))

        dof_states[..., 0] = humanoidverse_task.default_dof_pos

        dof_states[..., joint_ids, 0] = state["joint_pos"]
        dof_states[..., joint_ids, 1] = state["joint_vel"]

        ##
        root_states = torch.repeat_interleave(humanoidverse_task.base_init_state[None, :], \
                                              humanoidverse_task.num_envs, dim = 0)
        root_states[:, :3] = state["body_pos_w"][:, 0]
        base_rot = state["body_quat_w"][:, 0][:, [1, 2, 3, 0]] # wxyz -> xyzw
        root_states[:, 3:7] = base_rot

        root_states[:, 7:10] = state["body_lin_vel_w"][:, 0]
        root_states[:, 10:] = state["body_ang_vel_w"][:, 0]

        step_motion = {
            "dof_states": dof_states,
            "root_states": root_states
        }

        ###
        obs_dict, rew_buf, reset_buf, extras = humanoidverse_task.step_motions(step_motion)
        actor_obs = obs_dict['actor_obs']
        b = actor_obs.shape[0]

        body_pos = torch.reshape(state["local_body_pos"], (b, -1))
        body_quat = torch.reshape(state["local_body_quat"], (b, -1))
        lin_vel = torch.reshape(state["local_body_lin_vel"], (b, -1))
        ang_vel = torch.reshape(state["local_body_ang_vel"], (b, -1))

        _dim0 = 1
        _dim1 = _dim0 + body_pos.shape[1]
        diff_pos = actor_obs[:, _dim0: _dim1] - body_pos
        diff_pos2 = humanoidverse_task.lb_pos - body_pos

        _dim0 = _dim1
        _dim1 = _dim0 + body_quat.shape[1]
        diff_quat = actor_obs[:, _dim0: _dim1] - body_quat
        diff_quat2 = humanoidverse_task.tan_norm - body_quat

        _dim0 = _dim1
        _dim1 = _dim0 + lin_vel.shape[1]
        diff_lin_vel = actor_obs[:, _dim0: _dim1] - lin_vel
        diff_lin_vel2 = humanoidverse_task.lb_vel - lin_vel

        _dim0 = _dim1
        _dim1 = _dim0 + ang_vel.shape[1]
        diff_ang_vel = actor_obs[:, _dim0: _dim1] - ang_vel
        diff_ang_vel2 = humanoidverse_task.lb_ang_vel - ang_vel

        print(">>>>", torch.abs(diff_pos2).mean().cpu().item(), \
                      torch.abs(diff_quat2).mean().cpu().item(), \
                      torch.abs(diff_lin_vel2).mean().cpu().item(), \
                      torch.abs(diff_ang_vel2).mean().cpu().item())

        if reset_flag:
            humanoidverse_main.humanoidverse_final()
            return


if __name__ == "__main__":
    run_simulator(
        joint_names=[
            'left_hip_pitch_joint',
            'right_hip_pitch_joint',
            'waist_yaw_joint',
            'left_hip_roll_joint',
            'right_hip_roll_joint',
            'waist_roll_joint',
            'left_hip_yaw_joint',
            'right_hip_yaw_joint',
            'waist_pitch_joint',
            'left_knee_joint',
            'right_knee_joint',
            'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint',
            'left_ankle_pitch_joint',
            'right_ankle_pitch_joint',
            'left_shoulder_roll_joint',
            'right_shoulder_roll_joint',
            'left_ankle_roll_joint',
            'right_ankle_roll_joint',
            'left_shoulder_yaw_joint',
            'right_shoulder_yaw_joint',
            'left_elbow_joint',
            'right_elbow_joint',
            'left_wrist_roll_joint',
            'right_wrist_roll_joint',
            'left_wrist_pitch_joint',
            'right_wrist_pitch_joint',
            'left_wrist_yaw_joint',
            'right_wrist_yaw_joint'
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
