
import torch
import h5py
import numpy as np
from typing import Tuple
import collections
import tqdm
import numbers
from pathlib import Path
import copy

class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.input_dt = 1.0 / self.input_fps
        self.current_idx = 0
        self._load_motion()

    def _load_motion(self, keys = ["qpos", "qvel"]):
        hf = h5py.File(self.motion_file, "r")
        data = []
        num_ep = hf.attrs["num_episodes"]
        for i in range(num_ep):
            episode = hf[f"ep_{i}"]
            # ep = {k: torch.tensor(episode[k][:], dtype=torch.float32) for k in keys}
            ep = {k: copy.deepcopy(episode[k][:]) for k in episode.keys()}
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
        Tuple, bool
    ]:
        """Gets the next state of the motion."""

        state = {key:
            self.episode[key][self.current_idx : self.current_idx + 1] \
            for key in self.episode
        }

        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.input_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


import humenv
import mujoco_viewer

env = humenv.HumEnv()
env_dt = env.model.opt.timestep * env.action_repeat

viewer = mujoco_viewer.MujocoViewer(env.model, env.data)

motion = MotionLoader(
        motion_file="/workspace/ISAACSIM45ENVS/META/humenv/data_preparation/humenv_amass/0-ACCAD_Female1General_c3d_A2-Sway_poses.hdf5",
        input_fps=30
    )

observation, _ = env.reset()


while True:
    state, reset_flag = motion.get_next_state()
    motions = {
            "qpos": state["qpos"],
            "qvel": state["qvel"]
        }

    qpos = state["qpos"].flatten()
    qvel = state["qvel"].flatten()

    env.set_physics(qpos=qpos, qvel=qvel)
    observation = env.get_obs()["proprio"]
    viewer.render()

    #diff = state["observation"][0] - observation
    #print(diff)
