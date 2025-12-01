from time import time
from warnings import WarningMessage
import numpy as np
import os

import torch
from humanoidverse.utils.torch_utils import *
from humanoidverse.utils.spatial_utils import rotations
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase

from humanoidverse_env.misc import motionlib


class BFMTask(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self._init_motions()
        self.init_done = True

    def _init_motions(self):
        self.motions = motionlib.MotionBuffer(files=self.config.robot.asset.motions, base_path=self.config.robot.asset.motions_root)
        body_ids = []
        for name in self.config.robot.asset.mujoco_body_names:
            body_ids.append(self.body_names.index(name))
        self.mujoco_body_ids = torch.tensor(body_ids, dtype = torch.long, device = self.device)

        joint_ids = []
        for name in self.config.robot.asset.mujoco_joint_names:
            joint_ids.append(self.dof_names.index(name))

        self.mujoco_joint_ids = torch.tensor(joint_ids, dtype = torch.long, device = self.device)

    def step_motions(self, target_states):
        ## sample reset_all
        """ Reset all robots"""
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device), target_states)
        self.simulator.set_actor_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.dof_state)
        # self._refresh_env_idx_tensors(torch.arange(self.num_envs, device=self.device))
        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actor_state = {}
        actor_state["actions"] = actions
        return self.step(actor_state)

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # if target_states is not None, reset to target states
        if target_states is not None:
            if "dof_states" in target_states:
                self._reset_dofs(env_ids, target_states["dof_states"])
                self._reset_root_states(env_ids, target_states["root_states"])
            else:
                self._reset_motions(env_ids, target_states)
        else:
            prob = torch.tensor([self.config.robot.asset.motions_sample_ratio, 1 - self.config.robot.asset.motions_sample_ratio],
                                dtype=torch.float32, device=env_ids.device)

            mixidxs = torch.multinomial(prob, num_samples=env_ids.shape[0], replacement=True)
            env_ids = env_ids[mixidxs == 0]


            size = env_ids.shape[0]
            if size <= 0:
                return
            motions = self.motions.sample(size)
            self._reset_motions(env_ids, motions)

    def _reset_motions(self, env_ids, motions):
        size = env_ids.shape[0]
        qpos = torch.tensor(motions["qpos"], dtype = torch.float32, device = self.device)
        qvel = torch.tensor(motions["qvel"], dtype = torch.float32, device = self.device)

        dof_states = torch.zeros((size, self.num_dof, 2), device=str(self.device))
        dof_states[..., 0] = self.default_dof_pos

        dof_states[..., self.mujoco_joint_ids, 0] = qpos[:, 7:]
        dof_states[..., self.mujoco_joint_ids, 1] = qvel[:, 6:]
        self._reset_dofs(env_ids, dof_states)

        root_states = torch.repeat_interleave(self.base_init_state[None, :], size, dim = 0)
        root_states[:, :3] = qpos[:, 0:3]

        root_states[:, 3:7] = qpos[:, [4, 5, 6, 3]]  # wxyz -> xyzw

        root_states[:, 7:10] = qvel[:, :3]  ### lin_vel  qvel[:, :3]
        root_states[:, 10:] = qvel[:, 3:6]  ### ang_vel  qvel[:, 3:6]

        self._reset_root_states(env_ids, root_states)

    ######################### Observations #########################

    def _get_obs_root_h_obs(self):
        return self.simulator._rigid_body_pos[:, 0, 2:]

    def _get_obs_local_body_pos(self):
        root_body = self.simulator._rigid_body_pos[:, :1]
        body_pos_w = self.simulator._rigid_body_pos[:, self.mujoco_body_ids]
        lb_pos_w = body_pos_w - root_body
        lb_pos_w = lb_pos_w[:, 1:, :]

        body_size = self.mujoco_body_ids.shape[0] -1
        quats = torch.repeat_interleave(self.base_quat[:, None, :], body_size, dim = 1)
        quats = torch.reshape(quats, (-1, 4))

        lb_pos_w = torch.reshape(lb_pos_w, (-1, 3))
        lb_pos = quat_rotate_inverse(quats, lb_pos_w)
        lb_pos = lb_pos
        lb_pos = torch.reshape(lb_pos, (-1, body_size * 3))
        return lb_pos

    def _get_obs_local_body_rot_obs(self):
        body_size = self.mujoco_body_ids.shape[0]
        body_rot_w = self.simulator._rigid_body_rot[:, self.mujoco_body_ids]

        quats_inv = rotations.quat_inverse(self.base_quat, w_last = True)
        quats_inv = torch.repeat_interleave(quats_inv[:, None, :], body_size, dim = 1)

        body_rot_w = torch.reshape(body_rot_w, (-1, 4))
        quats_inv = torch.reshape(quats_inv, (-1, 4))

        local_body_rot = rotations.quat_mul_norm(quats_inv, body_rot_w, w_last=True)
        tan_norm = quat_to_tan_norm_xyzw(local_body_rot)

        tan_norm = torch.reshape(tan_norm, (-1, body_size * 6))
        return tan_norm

    def _get_obs_local_body_vel(self):
        body_vel_w = self.simulator._rigid_body_vel[:, self.mujoco_body_ids]

        body_size = self.mujoco_body_ids.shape[0]
        quats = torch.repeat_interleave(self.base_quat[:, None, :], body_size, dim = 1)
        quats = torch.reshape(quats, (-1, 4))
        body_vel_w = torch.reshape(body_vel_w, (-1, 3))
        lb_vel = quat_rotate_inverse(quats, body_vel_w)
        lb_vel = torch.reshape(lb_vel, (-1, body_size * 3))
        return lb_vel

    def _get_obs_local_body_ang_vel(self):
        body_ang_vel_w = self.simulator._rigid_body_ang_vel[:, self.mujoco_body_ids]

        body_size = self.mujoco_body_ids.shape[0]
        quats = torch.repeat_interleave(self.base_quat[:, None, :], body_size, dim = 1)
        quats = torch.reshape(quats, (-1, 4))
        body_ang_vel_w = torch.reshape(body_ang_vel_w, (-1, 3))
        lb_vel = quat_rotate_inverse(quats, body_ang_vel_w)
        lb_vel = torch.reshape(lb_vel, (-1, body_size * 3))
        return lb_vel




@torch.jit.script
def quat_to_tan_norm_xyzw(q):
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., :3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., :3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan
