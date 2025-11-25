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


@dataclasses.dataclass
class MotionBufferConfig():
    device: str = "cpu"

    motions_root: Optional[str] = None

    observations_key : list[str] = dataclasses.field(default_factory=list)
    privileges_key : list[str] = dataclasses.field(default_factory=list)


    fps: int = 50
    seq_length: int = 8
    history_horizon: int = 4

    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2


    def __post_init__(self):
        self.observations_key = ["base_ang_vel", "gravity", "joint_pos" ,"joint_vel"]
        self.privileges_key = [
            "base_lin_vel",
            "base_pos_z",
            "local_body_lin_vel",
            "local_body_ang_vel",
            "local_body_pos",
            "local_body_quat",
            "joint_acc",
            "joint_stiffness",
            "joint_damping",
            "friction_coeff",
            "torques",
            "feet_status",
            "feet_forces"
        ]

    def make_buffer(self):
        return MotionBuffer(self)


class MotionItems:
    slices: int = 0
    observations: Optional[torch.Tensor] = None
    privileges: Optional[torch.Tensor] = None

    def sample(self, cfg: MotionBufferConfig, output: defaultdict[list]):
        timeidx = torch.randint(0, self.slices, (1,))

        historys = []
        for idx in range(cfg.history_horizon):
            historys.append(timeidx + idx)

        history = torch.cat(historys)
        current = timeidx + cfg.history_horizon - 1
        # next = timeidx + cfg.history_horizon

        history_slices = []
        current_slices = []
        # next_slices = []

        for idx in range(cfg.seq_length):
            history_slices.append(history + idx)
            current_slices.append(current + idx)
            # next_slices.append(next + idx)

        history_slices = torch.cat(history_slices, dim = 0)
        current_slices = torch.cat(current_slices, dim = 0)
        # next_slices = torch.cat(next_slices, dim = 0)

        observations = self.observations[history_slices]
        observations = torch.reshape(observations, (cfg.seq_length, cfg.history_horizon, -1))
        # history_slices = torch.reshape(history_slices, (cfg.seq_length, cfg.history_horizon))
        privileges = self.privileges[current_slices]

        next_observations = self.observations[history_slices + 1]
        next_observations = torch.reshape(next_observations, (cfg.seq_length, cfg.history_horizon, -1))
        next_privileges = self.privileges[current_slices + 1]

        output["observations"].append(observations)
        output["privileges"].append(privileges)
        output["next"]["observations"].append(next_observations)
        output["next"]["privileges"].append(next_privileges)


class MotionBuffer:
    def __init__(self, cfg: MotionBufferConfig):
        self.config = cfg
        self.device = torch.device("cpu")

        self.capacity = 0
        self.storages = {}
        self.trajectory_priorities = []

        self._load_trajectories()

    def _load_file(self, motion_file):
        data = np.load(motion_file)

        slice_length = self.config.seq_length + self.config.history_horizon + 1

        observations = []
        for key in self.config.observations_key:
            observations.append(
                torch.tensor(data[key], dtype=torch.float32, device=self.device))

        privileges = []
        for key in self.config.privileges_key:
            privileges.append(
                torch.tensor(data[key], dtype=torch.float32, device=self.device))


        observations = torch.cat(observations, dim = -1)
        privileges = torch.cat(privileges, dim = -1)

        if len(observations.shape) == 3 and observations.shape[1] == 1:
            observations = torch.squeeze(observations, dim = 1)

        if len(privileges.shape) == 3 and privileges.shape[1] == 1:
            privileges = torch.squeeze(privileges, dim = 1)


        del data
        if observations.shape[0] < slice_length:

            del observations
            del privileges
            return

        slices = observations.shape[0] - slice_length
        items = MotionItems()
        items.slices = slices
        items.observations = observations
        items.privileges = privileges

        self.storages[self.capacity] =items
        self.trajectory_priorities.append(slices)
        self.capacity += 1

    def _load_trajectories(self):
        for root, dirs, files in os.walk(self.config.motions_root):
            for file in files:
                if not file.endswith(".npz"):
                    continue
                full_file = osp.join(root, file)
                self._load_file(full_file)

        self.trajectory_priorities = torch.tensor(self.trajectory_priorities, \
                                                  dtype= torch.float32, device = self.device)

        self.trajectory_priorities = self.trajectory_priorities.sqrt()
        self.trajectory_priorities /= torch.sum(self.trajectory_priorities)

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


    def priorities(self, metrics):
        pass

    def update_priorities(self, items):
        pass

