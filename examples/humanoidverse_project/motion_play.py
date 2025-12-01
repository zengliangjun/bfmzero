import os
import sys
import os.path as osp
root = osp.join(osp.dirname(__file__), "../..")

if root not in sys.path:
    sys.path.insert(0, root)

os.chdir(root)

from humanoidverse_env import main

main.humanoidverse_start()
humanoidverse_task = main.humanoidverse_task

import torch
import h5py
import numpy as np
from typing import Tuple
import collections
import tqdm
import numbers
from pathlib import Path


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        device: torch.device,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.input_dt = 1.0 / self.input_fps
        self.current_idx = 0
        self.device = device
        self._load_motion()

    def _load_motion(self, keys = ["qpos", "qvel"]):
        hf = h5py.File(self.motion_file, "r")
        data = []
        num_ep = hf.attrs["num_episodes"]
        for i in range(num_ep):
            episode = hf[f"ep_{i}"]
            # ep = {k: torch.tensor(episode[k][:], dtype=torch.float32) for k in keys}
            ep = {k: torch.tensor(episode[k][:], dtype=torch.float32) for k in episode.keys()}
            #ep["file_name"] = self.motion_file

            qpos = ep["qpos"]
            #self.qvel = ep["qvel"]

            self.input_frames = qpos.shape[0]
            self.duration = (self.input_frames - 1) * self.input_dt
            print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")
            self.episode = ep

    def get_next_state(
        self,
    ) -> Tuple[
        Tuple[
        torch.Tensor,
        torch.Tensor], bool
    ]:
        """Gets the next state of the motion."""

        state = {key:
            self.episode[key][self.current_idx : self.current_idx + 1].to(self.device) \
            for key in self.episode
        }

        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.input_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag

def run_simulator():
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file="/workspace/ISAACSIM45ENVS/META/humenv/data_preparation/humenv_amass/0-ACCAD_Female1General_c3d_A7-crouch_poses.hdf5",
        input_fps=30,
        device=humanoidverse_task.device,
    )
    num_envs: int = humanoidverse_task.config.num_envs
    algo_obs_dim_dict = humanoidverse_task.config.robot.algo_obs_dim_dict
    num_act = humanoidverse_task.config.robot.actions_dim

    obs_dict = humanoidverse_task.reset_all()
    while True:
        state, reset_flag = motion.get_next_state()
        motions = {
                "qpos": state["qpos"],
                "qvel": state["qvel"]
            }

        humanoidverse_task.step_motions(motions)

if __name__ == "__main__":
    run_simulator()
