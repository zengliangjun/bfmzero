import torch
import numpy as np
from typing import Tuple, Optional

class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        device: torch.device,
    ):
        frames = 0
        out_items = {}
        with open(motion_file, 'r+b') as fd:
            items = np.load(fd)

            for file in items.files:
                out_items[file] = torch.tensor(items[file], dtype=torch.float32, device=device)
                if 0 == frames:
                    frames = out_items[file].shape[0]

        self.current_idx = 0
        self.frames = frames
        self.out_items = out_items

    def get_next_state(
        self,
    ) -> Tuple[dict, bool]:

        """Gets the next state of the motion."""
        outs = {}
        for name in self.out_items:
            outs[name] = self.out_items[name][self.current_idx : self.current_idx + 1].clone()

        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.frames:
            self.current_idx = 0
            reset_flag = True

        return outs, reset_flag

