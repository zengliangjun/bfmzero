import torch
import dataclasses
from collections import defaultdict
from metamotivo.buffers.buffers import DictBuffer

@dataclasses.dataclass
class CollectConfig:
    # steps
    device: str = "cpu"
    seed_steps: int = 10
    observations_key : list = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.observations_key = ["actor_obs", "critic_obs", "history_obs", "history_actions"]

class CollectContext:

    cfg: CollectConfig
    collect_context: dict
    collect_buffers: dict

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def _build_observations(self, device):
        actor_obs = self.collect_buffers["actor_obs"]
        critic_obs = self.collect_buffers["critic_obs"]
        history_obs = self.collect_buffers["history_obs"]
        history_actions = self.collect_buffers["history_actions"]

        actor_obs =  torch.cat((history_obs, actor_obs[:, None, :]), dim = 1)

        train_status = {
            "obs": actor_obs.to(device),
            "action": history_actions.to(device),
            "privileges": critic_obs.to(device)
        }
        return train_status

    def _build_collect(self, collect: dict, output: defaultdict, indexes: torch.Tensor, device: str):
        actor_obs = collect["actor_obs"]
        critic_obs = collect["critic_obs"]
        history_obs = collect["history_obs"]
        history_actions = collect["history_actions"]

        actor_obs =  torch.cat((history_obs, actor_obs[:, None, :]), dim = 1)

        output["observations"] = actor_obs[indexes].clone().to(device)
        output["privileges"] = critic_obs[indexes].clone().to(device)
        output["history_action"] = history_actions[indexes].clone().to(device)

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
            self._build_collect(self.collect_buffers, output, indexes, self.cfg.device)

            next = defaultdict(list)
            next["rewards"] = rewards.to(self.cfg.device)[indexes][:, None]
            next["terminated"] = next_obs["terminated"].to(self.cfg.device)[indexes][:, None]
            self._build_collect(next_obs, next, indexes, self.cfg.device)
            output["next"] = next

        else:
            output = None

        ## reset done to 0
        if done is not None:
            for key in self.collect_buffers:
                self.collect_buffers[key][done, ...] = 0

        ## update buffers
        # for key, value in next_obs.items():
        for key in self.cfg.observations_key:
            value = next_obs[key]
            self.collect_buffers[key] = torch.zeros_like(value, device = self.cfg.device)
            self.collect_buffers[key][...] = value.to(self.cfg.device)
        return output

    def reset(self, env):
        obs, extras = env.reset()
        self.collect_context = {
                "obs": obs,
                'done': None,
                "context_z": None
            }

        self.collect_buffers = {}

        #for key, value in obs.items():
        for key in self.cfg.observations_key:
            value = obs[key]
            self.collect_buffers[key] = torch.zeros_like(value, device = self.cfg.device)
            self.collect_buffers[key][...] = value.to(self.cfg.device)


    def collect_step(self, step, env, agent, buffer: DictBuffer):
        data = self._train_collect_one_step(env, agent, step)
        if data is not None:
            buffer.extend(data)
