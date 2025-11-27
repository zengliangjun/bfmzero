import torch
import numpy as np
from loguru import logger as ulogger
import os.path as osp
from typing import Dict, Union, Tuple, Mapping, Optional, Any

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import dataclasses

from metamotivo.fb_bfmzero.isaac.parser_cfg import parse_env_cfg


@dataclasses.dataclass
class IsaacConfig():
    workdir: Optional[str] = None
    args_cli: Optional[Any] = None

    def make_env(self):
        return IsaacWarp(self)

class IsaacWarp:

    def __init__(self, cfg: IsaacConfig):
        self.config = cfg

        env_cfg = parse_env_cfg(
                cfg.args_cli.task,
                device=cfg.args_cli.device,
                num_envs=cfg.args_cli.num_envs,
                use_fabric=not cfg.args_cli.disable_fabric,
                entry_point_key="env_cfg_entry_point",
            )

        envs = gym.make(cfg.args_cli.task, cfg=env_cfg, render_mode="rgb_array" if cfg.args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(envs.unwrapped, DirectMARLEnv):
            envs = multi_agent_to_single_agent(envs)

        # wrap for video recording
        if cfg.args_cli.video:
            video_kwargs = {
                "video_folder": osp.join(cfg.workdir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": cfg.args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            envs = gym.wrappers.RecordVideo(envs, **video_kwargs)

        # wrap around environment for rsl-rl
        self.envs = RslRlVecEnvWrapper(envs)
        self.env_cfg = env_cfg

    def _calcute_objs(self, items):
        data = items["policy"]
        observations = {}
        for key, value in data.items():
            observations[key] = value.detach().clone()
        return observations


    def reset(self) -> Tuple[torch.Tensor, Dict[str, object]]:
        obs, infos = self.envs.get_observations()
        obs = self._calcute_objs(infos['observations'])
        return obs, infos

    def timestep(self) -> torch.Tensor:
        return self.envs.episode_length_buf

    def sample_action(self) -> torch.Tensor:
        actions = torch.randn((self.envs.num_envs, self.envs.num_actions), dtype = torch.float32, device=self.envs.device)
        return actions

    def step(self, actions: Union[np.ndarray, torch.Tensor]) -> \
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor, Dict[str, object]]:

        if actions.device != self.envs.device:
            actions = actions.to(self.envs.device)

        # clip actions
        if self.envs.clip_actions is not None:
            actions = torch.clamp(actions, -self.envs.clip_actions, self.envs.clip_actions)
        # record step information
        obs_dict, rewards, terminated, truncated, infos = self.envs.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs_dict = self._calcute_objs(obs_dict)
        obs_dict["terminated"] = terminated
        obs_dict["truncated"] = truncated

        # return the step information
        return obs_dict, rewards, dones, infos

    def end(self):
        pass

    def update_priorities(self, items):
        pass
