import torch
import dataclasses
from collections import defaultdict

from metamotivo.buffers.buffers import DictBuffer

@dataclasses.dataclass
class CollectConfig:
    # steps
    device: str = "cpu"
    seed_steps: int = 10
    history_horizon: int = 4
    action_key: str = "history_action"
    observations_key : list[str] = dataclasses.field(default_factory=list)
    privileges_key : list[str] = dataclasses.field(default_factory=list)


    def __post_init__(self):
        #self.observations_key = ["base_ang_vel", "gravity", "joint_pos" ,"joint_vel"]
        '''
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

class CollectContext:

    cfg: CollectConfig
    collect_context: dict
    collect_buffers: dict

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def _build_observations(self, device):
        observations = []
        for key in self.cfg.observations_key:
            observations.append(self.collect_buffers[key])

        action = self.collect_buffers[self.cfg.action_key]

        privileges = []
        for key in self.cfg.privileges_key:
            value: torch.Tensor = self.collect_buffers[key]
            if value.dim() == 3:
                value = value.reshape((value.shape[0], -1))
            privileges.append(value)

        train_status = {
            "obs": torch.cat(observations, dim = -1).to(device),
            "action": action.to(device),
            "privileges": torch.cat(privileges, dim = -1).to(device)
        }

        return train_status

    def _build_collect(self, output: defaultdict, indexes: torch.Tensor, device: str):
        for key in self.cfg.observations_key:
            output[key] = self.collect_buffers[key][indexes].clone().to(device)

        output[self.cfg.action_key] = self.collect_buffers[self.cfg.action_key][indexes].clone().to(device)

        for key in self.cfg.privileges_key:
            output[key] = self.collect_buffers[key][indexes].clone().to(device)

    def _build_next_observations(self, next_obs):
        observations = []
        for key in self.cfg.observations_key:
            observations.append(next_obs[key])

        privileges = []
        for key in self.cfg.privileges_key:
            privileges.append(next_obs[key])

        return torch.cat(observations, dim = -1), torch.cat(privileges, dim = -1)

    def _next_build_collect(self, output: defaultdict, obs: dict, indexes: torch.Tensor, device: str):
        for key in self.cfg.observations_key:
            output[key] = obs[key][indexes].clone().to(device)

        output[self.cfg.action_key] = obs[self.cfg.action_key][indexes].clone().to(device)

        for key in self.cfg.privileges_key:
            output[key] = obs[key][indexes].clone().to(device)

    @torch.no_grad()
    def _train_collect_one_step(self, env, agent, step):
        context_z = self.collect_context['context_z']
        done = self.collect_context['done']

        step_count = env.timestep()[:, None].to(agent.device)
        context_z = agent.maybe_update_rollout_context(z=context_z, step_count=step_count)

        if step < self.cfg.seed_steps:
            current_action = env.sample_action()
            obs = None
        else:
            # this works in inference mode
            obs = self._build_observations(agent.device)
            current_action = agent.act(obs=obs, z=context_z.to(agent.device), mean=False)

        next_obs, rewards, next_dones, next_infos = env.step(current_action)

        ## update context
        self.collect_context['context_z'] = context_z
        self.collect_context['done'] = next_dones

        if done is not None and done.shape[0] != torch.sum(done.float()):
            indexes = ~done
            indexes = indexes.to(self.cfg.device)

            output = defaultdict(list)

            output["action"] = current_action.to(self.cfg.device)[indexes]
            output["z"] = context_z.to(self.cfg.device)[indexes]
            self._build_collect(output, indexes, self.cfg.device)

            next = defaultdict(list)
            next["rewards"] = rewards.to(self.cfg.device)[indexes][:, None]
            next["terminated"] = next_obs["terminated"].to(self.cfg.device)[indexes][:, None]
            self._next_build_collect(next, next_obs, indexes, self.cfg.device)
            output["next"] = next

        else:
            output = None

        ## reset done to 0
        if done is not None:
            for key in self.collect_buffers:
                self.collect_buffers[key][done, ...] = 0

        ## update buffers
        for key, value in next_obs.items():
            if key in self.cfg.observations_key:
                self.collect_buffers[key][:, :-1, :] = self.collect_buffers[key][:, 1:, :].clone()
                self.collect_buffers[key][:, -1, :] = value.to(self.cfg.device)
            elif key in self.cfg.privileges_key:
                self.collect_buffers[key][...] = value.to(self.cfg.device)
            elif key == self.cfg.action_key:
                self.collect_buffers[key][:, :-1, :] = self.collect_buffers[key][:, 1:, :].clone()
                self.collect_buffers[key][:, -1, :] = value.to(self.cfg.device)

        return output

    def reset(self, env):
        obs, extras = env.reset()
        self.collect_context = {
                "obs": obs,
                'done': None,
                "context_z": None
            }

        self.collect_buffers = {}

        for key, value in obs.items():
            if key in self.cfg.observations_key:
                self.collect_buffers[key] = torch.zeros((value.shape[0], self.cfg.history_horizon + 1, *value.shape[1:]), device = self.cfg.device)

                self.collect_buffers[key][:, -1, :] = value.to(self.cfg.device)
            elif key in self.cfg.privileges_key:
                self.collect_buffers[key] = torch.zeros_like(value, device = self.cfg.device)
                self.collect_buffers[key][...] = value.to(self.cfg.device)
            elif key == self.cfg.action_key:
                self.collect_buffers[key] = torch.zeros((value.shape[0], self.cfg.history_horizon, *value.shape[1:]), device = self.cfg.device)
                self.collect_buffers[key][:, -1, :] = value.to(self.cfg.device)

    def collect_step(self, step, env, agent, buffer: DictBuffer):
        data = self._train_collect_one_step(env, agent, step)
        if data is not None:
            buffer.extend(data)
