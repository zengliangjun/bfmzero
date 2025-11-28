from __future__ import annotations
from typing import TYPE_CHECKING, Literal
import torch
from isaaclab.managers import ManagerTermBase, EventTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def randomize_gravity_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    gravity_range: tuple[float, float],
    curriculum_progress: float
):
    """根据课程进度随机化重力参数"""
    # 计算当前目标重力
    min_gravity, max_gravity = gravity_range
    current_gravity = min_gravity + (max_gravity - min_gravity) * curriculum_progress
    
    # 设置物理场景的重力
    env.sim.physics_sim_view.set_gravity((0.0, 0.0, current_gravity))


