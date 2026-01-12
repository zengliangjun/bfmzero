
import torch
from humanoidverse.utils.torch_utils import *
from humanoidverse.utils.spatial_utils import rotations
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse_env.motions import motions_buffer
from humanoidverse_env.isaac_utils import rotations

class MotionsTask(LeggedRobotBase):
    def __init__(self, config, device):
        super().__init__(config, device)

    def _setup_robot_body_indices(self):
        super()._setup_robot_body_indices()
        if hasattr(self.config.robot, "dof_ankle_roll_names"):
            self.ankle_roll_indices = [self.dof_names.index(dof) for dof in self.config.robot.dof_ankle_roll_names]

        if hasattr(self.config.robot.asset, "motions_root"):
            self.reset_motion_buffer = motions_buffer.MotionBuffer(self.config.robot.asset.motions_root)

        if hasattr(self.config.robot, "extend_config"):
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.extend_config:
                extend_parent_ids.append(self.simulator._body_list.index(extend_config["parent_name"]))
                # extend_parent_ids.append(self.simulator.find_rigid_body_indice(extend_config["parent_name"]))
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                self.simulator._body_list.append(extend_config["joint_name"])

            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.num_extend_bodies = len(extend_parent_ids)

            self.marker_coords = torch.zeros(self.num_envs,
                                         self.num_bodies + self.num_extend_bodies,
                                         3,
                                         dtype=torch.float,
                                         device=self.device,
                                         requires_grad=False) # extend

    def _init_buffers(self):
        super()._init_buffers()
        if hasattr(self.config.termination, "terminate_by_terminate_history_count") and \
            self.config.termination.terminate_by_terminate_history_count > 0:

            self.terminate_history_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)

    def _update_reset_buf(self):
        super()._update_reset_buf()

        if hasattr(self, "terminate_history_count"):
            terminated = (self.reset_buf != 0).to(torch.int32)
            self.terminate_history_count = self.terminate_history_count * terminated + self.reset_buf.to(torch.int32)
            self.reset_buf[:] = self.terminate_history_count >= self.config.termination.terminate_by_terminate_history_count

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        super().reset_envs_idx(env_ids, target_states, target_buf)
        if hasattr(self, "terminate_history_count") and len(env_ids) != 0:
            self.terminate_history_count[env_ids] = 0

        if hasattr(self, "reset_motion_buffer"):
            if self.common_step_counter % (self.max_episode_length * 1000) == 0:
                self.reset_motion_buffer.update_buffer()

    ######################### for motion play #########################
    def get_motion_joint(self, joint_names: list):
        joint_ids = []
        for name in joint_names:
            joint_ids.append(self.dof_names.index(name))
        return torch.tensor(joint_ids, dtype = torch.long, device = self.device)

    def get_motion_body(self, body_names: list):
        body_ids = []
        for name in body_names:
            body_ids.append(self.body_names.index(name))
        return torch.tensor(body_ids, dtype = torch.long, device = self.device)

    ######################### for motion play #########################
    def step_motions(self, motions_states):
        setattr(self, "motions_states", motions_states)
        self.motions_states = motions_states
        self.render()
        for _ in range(self.config.simulator.config.sim.control_decimation):
            self.simulator.simulate_at_each_physics_step()

        self._post_physics_step()
        delattr(self, "motions_states")
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras


    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        if hasattr(self, "motions_states") and self.motions_states != None:
            reset_env_ids = torch.arange(self.num_envs, device=self.device)
            super().reset_envs_idx(reset_env_ids, self.motions_states)
        elif hasattr(self, "reset_motion_buffer"):
            self._reset_envs_idx_with_motion_buffer(env_ids, target_states=target_states, target_buf=target_buf)
        else:
            super().reset_envs_idx(env_ids, target_states, target_buf)

    def _reset_envs_idx_with_motion_buffer(self, env_ids, target_states=None, target_buf=None):
        size = len(env_ids)
        if len(env_ids) == 0:
            return

        dof_states = torch.zeros((size, self.num_dof, 2), device=str(self.device))
        dof_states[:, :, 0] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (size, self.num_dof), device=str(self.device))

        root_states = torch.repeat_interleave(self.base_init_state[None, :], size, dim = 0)
        root_states[:, 7:13] = torch_rand_float(-0.5, 0.5, (size, 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel

        ##
        prob = torch.tensor([self.config.robot.asset.motions_sample_ratio, 1 - self.config.robot.asset.motions_sample_ratio],
                                dtype=torch.float32, device=env_ids.device)

        mixidxs = torch.multinomial(prob, num_samples=env_ids.shape[0], replacement=True)
        motions_sample = mixidxs == 0

        size = torch.sum(motions_sample.float()).cpu().item()
        if size != 0:
            motions = self.reset_motion_buffer.sample(size, self.device)
            dof_states[motions_sample, :] = motions['dof_states']
            root_states[motions_sample, :] = motions['root_states']

        target_states = {
            'dof_states': dof_states,
            'root_states': root_states,
        }
        super().reset_envs_idx(env_ids, target_states, target_buf)

    ######################### Observations #########################
    def calcute_with_extends(self):
        if not hasattr(self, "extend_body_parent_ids"):
            return
        rotated_pos_in_parent = rotations.my_quat_rotate(
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3)
        )
        extend_pos = rotations.my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent
        ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
        ##
        extend_rot = rotations.quat_mul(self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                                    self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                    w_last=True).view(self.num_envs, -1, 4)
        extend_ang_vel = self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]

        extend_ang_vel_contribution = torch.cross(extend_ang_vel, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
        extend_vel = self.simulator._rigid_body_vel[:, self.extend_body_parent_ids] + extend_ang_vel_contribution.view(self.num_envs, -1, 3)

        self.extend_rigid_body_pos = torch.cat((self.simulator._rigid_body_pos, extend_pos), dim=1)
        self.extend_rigid_body_rot = torch.cat((self.simulator._rigid_body_rot, extend_rot), dim=1)
        self.extend_rigid_ang_vel = torch.cat((self.simulator._rigid_body_ang_vel, extend_ang_vel), dim=1)
        self.extend_rigid_vel = torch.cat((self.simulator._rigid_body_vel, extend_vel), dim=1)


    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()

        ##
        if not hasattr(self, "extend_body_parent_ids"):
            rigid_body_pos = self.simulator._rigid_body_pos
            rigid_body_rot = self.simulator._rigid_body_rot
            rigid_body_ang_vel = self.simulator._rigid_body_ang_vel
            rigid_body_vel = self.simulator._rigid_body_vel
            num_bodies = self.num_bodies
        else:
            self.calcute_with_extends()
            rigid_body_pos = self.extend_rigid_body_pos
            rigid_body_rot = self.extend_rigid_body_rot
            rigid_body_ang_vel = self.extend_rigid_ang_vel
            rigid_body_vel = self.extend_rigid_vel
            num_bodies = self.num_bodies + self.num_extend_bodies

        heading_quats = rotations.calc_heading_quat(self.base_quat, w_last=True)
        heading_inv_quats = rotations.calc_heading_quat_inv(self.base_quat, w_last=True)

        heading_quats = torch.repeat_interleave(heading_quats[:, None, :], num_bodies, dim = 1)
        heading_quats = torch.reshape(heading_quats, (-1, 4))

        heading_inv_quats = torch.repeat_interleave(heading_inv_quats[:, None, :], num_bodies, dim = 1)
        heading_inv_quats = torch.reshape(heading_inv_quats, (-1, 4))



        # lb pos
        lb_pos_w = rigid_body_pos - rigid_body_pos[:, :1]
        lb_pos_w = torch.reshape(lb_pos_w, (-1, 3))
        lb_heading_pos = quat_rotate_inverse(heading_inv_quats, lb_pos_w)
        self.lb_heading_pos = torch.reshape(lb_heading_pos, (-1, num_bodies, 3))

        # body rot
        body_rot_w = rigid_body_rot
        body_rot_w = torch.reshape(body_rot_w, (-1, 4))
        heading_body_rot = rotations.quat_mul_norm(heading_inv_quats, body_rot_w, w_last=True)
        heading_body_tan_norm = quat_to_tan_norm_xyzw(heading_body_rot)
        self.lb_heading_tan_norm = torch.reshape(heading_body_tan_norm, (-1, num_bodies, 6))

        # vel
        body_vel_w = rigid_body_vel
        body_vel_w = torch.reshape(body_vel_w, (-1, 3))
        lb_heading_vel = quat_rotate_inverse(heading_inv_quats, body_vel_w)
        self.lb_heading_vel = torch.reshape(lb_heading_vel, (-1, num_bodies, 3))
        # ang vel
        body_ang_vel_w = rigid_body_ang_vel
        body_ang_vel_w = torch.reshape(body_ang_vel_w, (-1, 3))
        lb_heading_ang_vel = quat_rotate_inverse(heading_inv_quats, body_ang_vel_w)
        self.lb_heading_ang_vel = torch.reshape(lb_heading_ang_vel, (-1, num_bodies, 3))

    def _get_obs_root_h_obs(self):
        self.root_h_obs = self.simulator._rigid_body_pos[:, 0, 2:]
        return self.root_h_obs

    def _get_obs_local_body_pos(self):
        if hasattr(self, "motion_body_ids"):
            lb_heading_pos = self.lb_heading_pos[:, self.motion_body_ids]
        else:
            lb_heading_pos = self.lb_heading_pos[:, 1:]

        lb_heading_pos = torch.reshape(lb_heading_pos, (lb_heading_pos.shape[0], -1))
        return lb_heading_pos

    def _get_obs_local_body_rot_obs(self):
        if hasattr(self, "motion_body_ids"):
            lb_heading_tan_norm = self.lb_heading_tan_norm[:, self.motion_body_ids]
        else:
            lb_heading_tan_norm = self.lb_heading_tan_norm

        return torch.reshape(lb_heading_tan_norm, (lb_heading_tan_norm.shape[0], -1))

    def _get_obs_local_body_vel(self):
        if hasattr(self, "motion_body_ids"):
            body_vel_w = self.lb_heading_vel[:, self.motion_body_ids]
        else:
            body_vel_w = self.lb_heading_vel

        return torch.reshape(body_vel_w, (body_vel_w.shape[0], -1))

    def _get_obs_local_body_ang_vel(self):
        if hasattr(self, "motion_body_ids"):
            lb_heading_ang_vel = self.lb_heading_ang_vel[:, self.motion_body_ids]
        else:
            lb_heading_ang_vel = self.lb_heading_ang_vel

        return torch.reshape(lb_heading_ang_vel, (lb_heading_ang_vel.shape[0], -1))

    def _get_obs_history_obs(self,):
        assert "history_obs" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_obs']
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            # history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=-1)

    def _get_obs_history_actions(self,):
        assert "history_actions" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actions']
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            # history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=-1)

    ######################### REWARD #########################
    def _reward_penalty_ankle_roll(self):
        # Penalize dof positions too close to the limit
        diff = self.simulator.dof_pos[:, self.ankle_roll_indices] - self.default_dof_pos[:, self.ankle_roll_indices]
        return torch.sum(torch.abs(diff), dim=1)

    def _reward_penalty_contact(self):
        contacted = torch.norm(self.simulator.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1
        return torch.sum(contacted, dim=1)

    def _reward_penalty_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5


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


class MotionsTaskExt(MotionsTask):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._init_motions()

    def _init_motions(self):
        if hasattr(self.config.robot.asset, "motion_joint_names"):
            self.set_motion_joint(self.config.robot.asset.motion_joint_names)

        if hasattr(self.config.robot.asset, "motion_body_names"):
            self.set_motion_body(self.config.robot.asset.motion_body_names)
