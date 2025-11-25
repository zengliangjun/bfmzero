from __future__ import annotations

from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import isaaclab.utils.math as isaac_math_utils
from metamotivo.fb_bfmzero.isaac.mdps.events.motion_loader import MotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg

class StateInit(ManagerTermBase):

    cfg: EventTermCfg

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        motions_root = cfg.params.get("motions_root")
        self.motion_loader = MotionLoader(motions_root)

    def _reset_by_motions(self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg):

        asset: Articulation = env.scene[asset_cfg.name]

        root_states = asset.data.default_root_state[env_ids].clone()

        states = self.motion_loader.sample_states(root_states.shape[0], root_states.device)

        root_states[:, 0:3] +=  env.scene.env_origins[env_ids]
        root_states[:, 3:] = states["root_state"]

        asset.write_root_state_to_sim(root_states, env_ids = env_ids)
        asset.write_joint_state_to_sim(states["joint_pos"], states["joint_vel"], env_ids = env_ids)


    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        motions_root: str,
        motions_sample_ratio: float = 0.8
    ) -> torch.Tensor:

        prob = torch.tensor([motions_sample_ratio, 1 - motions_sample_ratio],
                            dtype=torch.float32, device=env_ids.device)

        mixidxs = torch.multinomial(prob, num_samples=env_ids.shape[0], replacement=True)
        env_motions_ids = env_ids[mixidxs == 0]

        self._reset_by_motions(env, env_motions_ids, asset_cfg)


