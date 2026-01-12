import os
import os.path as osp
import torch
from typing import Union

class MotionBuffer:
    def __init__(self, motions_root, device = torch.device("cpu")):
        self.motions_root = motions_root
        self.device = device

        full_files = []
        for root, dirs, files in os.walk(self.motions_root):
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
        if hasattr(self, "dof_states"):
            del self.dof_states
            del self.root_states

        self.dof_states = []
        self.root_states = []
        self._load_trajectories(self.full_files[self.current_file_idx])
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
        dof_states = data["dof_states"]
        root_states = data["root_states"]
        self.dof_states.append(dof_states)
        self.root_states.append(root_states)


    def _load_trajectories(self, motion_file):
        self._load_file(motion_file)
        self.dof_states = torch.cat(self.dof_states, dim = 0)
        self.root_states = torch.cat(self.root_states, dim = 0)

    def __len__(self) -> int:
        return self.dof_states.shape[0]

    def sample(self, batch_size, device: Union[torch.device, str]=None) -> torch.Tensor:
        ids = torch.randint(0, len(self), (int(batch_size), ))

        dof_states = self.dof_states[ids].clone().to(device)
        root_states = self.root_states[ids].clone().to(device)

        return {
            "dof_states": dof_states,
            "root_states": root_states,
        }

