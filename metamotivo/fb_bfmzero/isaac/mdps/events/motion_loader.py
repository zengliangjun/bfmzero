import os
import os.path as osp
import numpy as np
import torch
from typing import Sequence


class MotionLoader:
    def __init__(self, motions_root: str, ):
        self.motions_root = motions_root
        self.joint_pos = []
        self.joint_vel = []

        self.root_pos_w = []
        self.root_quat_w = []
        self.root_lin_vel_w = []
        self.root_ang_vel_w = []

        self._load_trajectories()
        self.joint_pos = torch.cat(self.joint_pos, dim = 0)
        self.joint_vel = torch.cat(self.joint_vel, dim = 0)

        self.root_pos_w = torch.cat(self.root_pos_w, dim = 0)
        self.root_quat_w = torch.cat(self.root_quat_w, dim = 0)
        self.root_lin_vel_w = torch.cat(self.root_lin_vel_w, dim = 0)
        self.root_ang_vel_w = torch.cat(self.root_ang_vel_w, dim = 0)
        self.size = self.joint_pos.shape[0]

    def _load_file(self, motion_file):
        data = np.load(motion_file, allow_pickle=True)

        file = data.files[0]
        if file.endswith(".npz") or file.endswith(".npy"):
            for file in data.files:
                item = data[file].item()
                self._load_data(item)
        else:
            self._load_data(data)

    def _load_data(self, data):
        self.joint_pos.append(torch.tensor(data["joint_pos"], dtype=torch.float32))
        self.joint_vel.append(torch.tensor(data["joint_vel"], dtype=torch.float32))

        self.root_pos_w.append(torch.tensor(data["body_pos_w"][:, 0], dtype=torch.float32))
        self.root_quat_w.append(torch.tensor(data["body_quat_w"][:, 0], dtype=torch.float32))
        self.root_lin_vel_w.append(torch.tensor(data["body_lin_vel_w"][:, 0], dtype=torch.float32))
        self.root_ang_vel_w.append(torch.tensor(data["body_ang_vel_w"][:, 0], dtype=torch.float32))

    def _load_trajectories(self):
        for root, dirs, files in os.walk(self.motions_root):
            for file in files:
                if not file.endswith(".npz") and not file.endswith(".npy"):
                    continue
                full_file = osp.join(root, file)
                self._load_file(full_file)

    def sample_states(self, size, device):
        sample_idx = torch.randint(0, self.size - 1, (size,))

        joint_pos = self.joint_pos[sample_idx].to(device)
        joint_vel = self.joint_vel[sample_idx].to(device)

        root_quat_w = self.root_quat_w[sample_idx].to(device)
        root_lin_vel_w = self.root_lin_vel_w[sample_idx].to(device)
        root_ang_vel_w = self.root_ang_vel_w[sample_idx].to(device)

        return {
            "root_state": torch.cat((root_quat_w, root_lin_vel_w, root_ang_vel_w), dim = 1),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }
