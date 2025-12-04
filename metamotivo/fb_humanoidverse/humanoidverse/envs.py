import torch
import numpy as np
from loguru import logger as ulogger
from typing import Tuple, Dict, Union
from humanoidverse.envs.legged_base_task import legged_robot_base

from humanoidverse_env import motions_main

class MotionsWarp:
    task: legged_robot_base.LeggedRobotBase
    def __init__(self):
        self.task = motions_main.humanoidverse_task

    def reset(self) -> Tuple[torch.Tensor, Dict[str, object]]:
        obs = self.task.reset_all()
        return obs, {}

    def timestep(self) -> torch.Tensor:
        return self.task.last_episode_length_buf

    def sample_action(self) -> torch.Tensor:
        actions = torch.randn((self.task.num_envs, self.task.dim_actions), dtype = torch.float32, device=self.task.device)
        return actions

    def step(self, actions: Union[np.ndarray, torch.Tensor]) -> \
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor, Dict[str, object]]:

        if actions.device != self.task.device:
            actions = actions.to(self.task.device)

        # record step information
        actor_state = {"actions": actions}
        obs_dict, rewards, reset_buf, infos = self.task.step(actor_state)
        # compute dones for compatibility with RSL-RL
        dones = reset_buf == 1
        no_time_out = torch.logical_not(self.task.time_out_buf)
        terminated = torch.logical_and(dones, no_time_out)

        # move extra observations to the extras dict
        obs_dict["terminated"] = terminated
        obs_dict["truncated"] = self.task.time_out_buf

        # return the step information
        return obs_dict, rewards, dones, infos

    def end(self):
        motions_main.humanoidverse_final()

    def update_priorities(self, items):
        pass

    def update_curriculum(self, progress: float):
        pass

