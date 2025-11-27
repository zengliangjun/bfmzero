
from typing import Dict
from metamotivo.buffers.buffers import DictBuffer

import torch
import dataclasses
from collections import defaultdict

@dataclasses.dataclass
class ReplayBuffer(DictBuffer):

    action_key: str = "history_action"
    observations_key : list[str] = dataclasses.field(default_factory=list)
    privileges_key : list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        '''
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
        '''
        ## setup with workspace.py
        pass

    def _build_observations(self, items):

        output = defaultdict(list)

        def _build_data(_items):
            observations = []
            for key in self.observations_key:
                observations.append(_items[key])

            privileges = []
            for key in self.privileges_key:
                value: torch.Tensor = _items[key]
                if len(value.shape) >= 3:
                    value = value.reshape((value.shape[0], -1))
                privileges.append(value)

            return torch.cat(observations, dim = -1), torch.cat(privileges, dim = -1)

        history_observations, privileges = _build_data(items)
        history_actions = items[self.action_key]

        output["observations"] = history_observations
        output["privileges"] = privileges
        output[self.action_key] = history_actions

        output["action"] = items["action"]
        output["z"] = items["z"]

        ##  next
        next = items["next"]
        next_observations, next_privileges = _build_data(next)

        next_output = defaultdict(list)

        next_output["observations"] = torch.cat((history_observations[:, 1:, :], next_observations[:, None, :]), dim = -2)
        next_output["privileges"] = next_privileges
        next_output[self.action_key] = torch.cat((history_actions[:, 1:, :], items["action"][:, None, :]), dim = -2)

        next_output["terminated"] = next["terminated"]
        next_output["rewards"] = next["rewards"]

        output["next"] = next_output

        return output

    def sample(self, batch_size) -> Dict[str, torch.Tensor]:
        items = super().sample(batch_size)
        return self._build_observations(items)
