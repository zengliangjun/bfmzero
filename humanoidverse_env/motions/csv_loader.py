import torch
from  humanoidverse_env.isaac_utils.isaaclab_math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

import pickle
import numpy as np
from typing import Tuple, Optional

class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: Tuple[int, int] = None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""

        if self.motion_file.endswith("csv"):
            if self.frame_range is None:
                motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
            else:
                motion = torch.from_numpy(
                    np.loadtxt(
                        self.motion_file,
                        delimiter=",",
                        skiprows=self.frame_range[0] - 1,
                        max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                    )
                )
            motion = motion.to(torch.float32).to(self.device)
            self.motion_base_poss_input = motion[:, :3]
            self.motion_base_rots_input = motion[:, 3:7]
            self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
            self.motion_dof_poss_input = motion[:, 7:]


        elif self.motion_file.endswith("pkl"):
            with open(self.motion_file, "rb") as f:
                motion_data = pickle.load(f)
            root_pos = motion_data["root_pos"]
            root_rot = motion_data["root_rot"]  #  [:, [3, 0, 1, 2]]  # xyzw → wxyz
            root_rot = root_rot[:, [3, 0, 1, 2]]  # convert to wxyz
            dof_pos = motion_data["dof_pos"]

            self.motion_base_poss_input = torch.from_numpy(root_pos).to(torch.float32).to(self.device)
            self.motion_base_rots_input = torch.from_numpy(root_rot).to(torch.float32).to(self.device)
            self.motion_dof_poss_input = torch.from_numpy(dof_pos).to(torch.float32).to(self.device)

            self.input_fps = motion_data["fps"]
            self.input_dt = 1.0 / self.input_fps


        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> Tuple[dict, bool]:

        """Gets the next state of the motion."""
        state = {
            "base_pos": self.motion_base_poss[self.current_idx : self.current_idx + 1],
            "base_rot": self.motion_base_rots[self.current_idx : self.current_idx + 1],
            "base_lin_vel": self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            "base_ang_vel": self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            "dof_pos": self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            "dof_vel": self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        }
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag

