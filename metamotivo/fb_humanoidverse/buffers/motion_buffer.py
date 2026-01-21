import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import torch
from collections import defaultdict
from typing import Union, Mapping, Optional, Dict
from pathlib import Path
from loguru import logger as ulogger
import dataclasses
import random

@dataclasses.dataclass
class MotionBufferConfig():
    device: str = "cpu"

    motions_root: Optional[str] = None

    fps: int = 50
    seq_length: int = 8
    slice_num_seconds: int = 4  # 4s

    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2

    def make_buffer(self):
        return MotionBuffer(self)

class MotionItems:
    slices: int = 0
    actor_obs: Optional[torch.Tensor] = None
    critic_obs: Optional[torch.Tensor] = None
    history_obs: Optional[torch.Tensor] = None

    def sample(self, cfg: MotionBufferConfig, output: defaultdict):
        timeidx = torch.randint(0, self.slices, (1,))

        slices = []
        for idx in range(cfg.seq_length):
            slices.append(timeidx + idx)

        slices = torch.cat(slices, dim = 0)
        next_slices = slices + 1

        actor = self.actor_obs[slices]
        critic = self.critic_obs[slices]
        history = self.history_obs[slices]

        observations =  torch.cat((history, actor[:, None, :]), dim = 1)

        next_actor = self.actor_obs[next_slices]
        next_critic = self.critic_obs[next_slices]
        next_history = self.history_obs[next_slices]

        next_observations =  torch.cat((next_history, next_actor[:, None, :]), dim = 1)

        output["observations"].append(observations)
        output["privileges"].append(critic)
        output["next"]["observations"].append(next_observations)
        output["next"]["privileges"].append(next_critic)

    def trajectories(self, output: defaultdict):
        observations =  torch.cat((self.history_obs, self.actor_obs[:, None, :]), dim = 1)
        output["observations"].append(observations)
        output["privileges"].append(self.critic_obs)

class MotionBuffer:
    def __init__(self, cfg: MotionBufferConfig):
        self.config = cfg
        self.device = torch.device("cpu")

        full_files = []
        for root, dirs, files in os.walk(self.config.motions_root):
            for file in files:
                if not file.endswith(".pth"):
                    continue
                full_file = osp.join(root, file)
                full_files.append(full_file)

        self.full_files = full_files
        import random
        random.shuffle(self.full_files)
        random.shuffle(self.full_files)
        self.current_file_idx = 0
        self.reload_buffer()

    def reload_buffer(self):
        if hasattr(self, "storages"):
            old_storages = self.storages
        else:
            old_storages = None

        self.capacity = 0
        self.storages = {}
        self.trajectory_priorities = []

        self._load_trajectories(self.full_files[self.current_file_idx])

        if old_storages:
            keys = list(old_storages.keys())
            keep_keys = random.sample(keys, len(keys) // 4)
            keep_storages = {k: old_storages[k] for k in keep_keys}
            self.storages.update(keep_storages)

        new_key = 0
        new_storages = {}
        for k, v in self.storages.items():
            new_storages[new_key] = v
            new_key += 1

        self.storages = new_storages
        self.trajectory_priorities = torch.ones((new_key), \
                                                  dtype= torch.float32, device = self.device)

        self.current_file_idx += 1
        self.current_file_idx %= len(self.full_files)


    def _load_file(self, motion_file):
        data = torch.load(motion_file)
        if "dof_states" in data:
            self._load_data(data)
        else:
            for key, value in data.items():
                self._load_data(value)

    def _load_data(self, data):
        slice_length = self.config.seq_length + 1

        slice_frames = self.config.fps  * self.config.slice_num_seconds

        actor_obs = data["actor_obs"]
        critic_obs = data["critic_obs"]
        history_obs = data["history_obs"]

        num_frames = actor_obs.shape[0]
        num_slices = num_frames // slice_frames

        for slices in range(num_slices):
            end_idx = min((slices + 1) * slice_frames, num_frames)
            start_idx = max(end_idx - slice_frames, 0)

            slices = (end_idx - start_idx) - slice_length
            items = MotionItems()
            items.slices = slices
            items.actor_obs = actor_obs[start_idx: end_idx].clone()
            items.critic_obs = critic_obs[start_idx: end_idx].clone()
            items.history_obs = history_obs[start_idx: end_idx].clone()

            self.storages[self.capacity] =items
            self.trajectory_priorities.append(slices)
            self.capacity += 1

    def _load_trajectories(self, full_file):
        self._load_file(full_file)

    def __len__(self) -> int:
        return len(self.capacity)

    def empty(self) -> bool:
        return self.capacity == 0

    def extend(self, data: torch.Tensor) -> None:
        raise Exception("Don\'t support extend")

    def sample_with_epinds(self, batch_size, device: Union[torch.device, str]=None) -> torch.Tensor:
        def _dict_cat(d: Mapping) -> Dict[str, torch.Tensor]:
            res = {}
            for k, v in d.items():
                if isinstance(v, Mapping):
                    res[k] = _dict_cat(v)
                else:
                    res[k] = torch.cat(v, dim=0)
            return res

        def recursive_to_device(_s, _d, _device):
            for k, v in _s.items():
                if isinstance(v, Mapping):
                    _dv = {}
                    recursive_to_device(v, _dv, _device)
                    _d[k]= _dv
                else:
                    _d[k]= v.to(_device)

        if device is None:
            device = self.config.device

        ##
        assert batch_size >= self.config.seq_length
        assert batch_size % self.config.seq_length == 0

        num_slices = batch_size // self.config.seq_length

        epinds = torch.multinomial(self.trajectory_priorities, num_slices, replacement=True)

        output = defaultdict(list)
        output["next"] = defaultdict(list)

        for epidx in epinds:
            _ep: MotionItems = self.storages[epidx.item()]
            _ep.sample(self.config, output)

        output = _dict_cat(output)

        results = {}
        recursive_to_device(output, results, device)

        return epinds, results

    def sample(self, batch_size, device: Union[torch.device, str]=None) -> torch.Tensor:
        _, output = self.sample_with_epinds(batch_size, device)
        return output

    def tracking_motions(self, num_envs):
        epinds = torch.multinomial(self.trajectory_priorities, num_envs, replacement=True)

        output = defaultdict(list)
        for epidx in epinds:
            _ep: MotionItems = self.storages[epidx.item()]
            _ep.trajectories(output)
        return output

    def priorities(self, metrics):
        pass

    def update_priorities(self, items):
        pass

if __name__ == "__main__":
    cfg = MotionBufferConfig()
    cfg.motions_root = "/workspace/data/motions"

    buffer = cfg.make_buffer()

    items = buffer.sample(64)
    for key, value in items.items():
        print(key, value)
