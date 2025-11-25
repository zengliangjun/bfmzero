
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject

def body_coms(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    rigid_body_mass = ObsTerm(
            func=body_coms,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: Articulation = env.scene[asset_cfg.name]
    coms = asset.data.com_pos_b[:, asset_cfg.body_ids]
    return coms.flatten(1)

def body_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    rigid_body_mass = ObsTerm(
            func=body_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: RigidObject = env.scene[asset_cfg.name]
    scales = asset.root_physx_view.get_masses().to(env.device) / asset.data.default_mass.to(env.device)
    return scales[:, asset_cfg.body_ids]

def push_force(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    rigid_body_mass = ObsTerm(
            func=push_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.root_vel_w[:, :2]

def push_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    rigid_body_mass = ObsTerm(
            func=push_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w

def local_body_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root link linear velocity in base frame. Shape is (num_instances, 3).

    This quantity is the linear velocity of the articulation root's actor frame with respect to the
    its actor frame.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    lin_vel_w_ref_b = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    quat_w = torch.repeat_interleave(asset.data.root_quat_w[:, None, :], lin_vel_w_ref_b.shape[1], dim=1)

    return math_utils.quat_apply_inverse(quat_w, lin_vel_w_ref_b).reshape(quat_w.shape[0], -1)

def local_body_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root link angular velocity in base world frame. Shape is (num_instances, 3).

    This quantity is the angular velocity of the articulation root's actor frame with respect to the
    its actor frame.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    quat_w = torch.repeat_interleave(asset.data.root_quat_w[:, None, :], body_ang_vel_w.shape[1], dim=1)
    return math_utils.quat_apply_inverse(quat_w, body_ang_vel_w).reshape(quat_w.shape[0], -1)

def local_body_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root link linear velocity in base frame. Shape is (num_instances, 3).

    This quantity is the linear velocity of the articulation root's actor frame with respect to the
    its actor frame.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    root_pos_w = asset.data.root_pos_w[:, None, :]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    local_pos_w = body_pos_w - root_pos_w

    quat_w = torch.repeat_interleave(asset.data.root_quat_w[:, None, :], local_pos_w.shape[1], dim=1)
    return math_utils.quat_apply_inverse(quat_w, local_pos_w).reshape(quat_w.shape[0], -1)

def local_body_quat(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]

    inv_quat_w = math_utils.quat_inv(asset.data.root_quat_w)
    inv_quat_w = torch.repeat_interleave(inv_quat_w[:, None, :], body_quat_w.shape[1], dim=1)

    lb_quat = math_utils.quat_mul(inv_quat_w, body_quat_w)
    lb_quat = math_utils.normalize(lb_quat)

    lb_mat = math_utils.matrix_from_quat(lb_quat)[..., :2]
    return lb_mat.reshape(lb_mat.shape[0], -1)
