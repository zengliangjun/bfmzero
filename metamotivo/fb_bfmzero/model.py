
import dataclasses
from torch import nn
import torch
import copy
import torch.nn.functional as F
import copy
from pathlib import Path
from safetensors.torch import save_model as safetensors_save_model
import json
import math
import numpy as np


from ..nn_models import build_forward, build_backward, build_discriminator, build_actor, eval_mode
from .. import config_from_dict, load_model

@dataclasses.dataclass
class ForwardCriticArchiConfig:
    hidden_dim: int = 2048
    model: str = "residual"  # {'simple', 'residual'}
    hidden_layers: int = 6
    embedding_layers: int = 4
    num_parallel: int = 2
    ensemble_mode: str = "batch"  # {'batch', 'seq', 'vmap'}


@dataclasses.dataclass
class DiscCriticArchiConfig(ForwardCriticArchiConfig):
    pass


@dataclasses.dataclass
class AuxiCriticArchiConfig(ForwardCriticArchiConfig):
    pass


@dataclasses.dataclass
class ActorArchiConfig:
    hidden_dim: int = 2048
    model: str = "residual"  # {'simple', 'residual'}
    hidden_layers: int = 6
    embedding_layers: int = 4


@dataclasses.dataclass
class DiscriminatorArchiConfig:
    hidden_dim: int = 1024
    hidden_layers: int = 2


@dataclasses.dataclass
class BackwardArchiConfig:
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm: bool = True



@dataclasses.dataclass
class ArchiConfig:
    z_dim: int = 256
    norm_z: bool = True
    f: ForwardCriticArchiConfig = dataclasses.field(default_factory=ForwardCriticArchiConfig)
    b: BackwardArchiConfig = dataclasses.field(default_factory=BackwardArchiConfig)
    actor: ActorArchiConfig = dataclasses.field(default_factory=ActorArchiConfig)

    critic: DiscCriticArchiConfig = dataclasses.field(default_factory=DiscCriticArchiConfig)
    discriminator: DiscriminatorArchiConfig = dataclasses.field(default_factory=DiscriminatorArchiConfig)

    auxi_critic: AuxiCriticArchiConfig = dataclasses.field(default_factory=AuxiCriticArchiConfig)


@dataclasses.dataclass
class Config():
    obs_dim: int = 64
    obs_history_horizon: int = 4
    obs_privileges_dim: int = 607

    action_dim: int = 29
    device: str = "cpu"
    inference_batch_size: int = 500_000
    seq_length: int = 8
    actor_std: float = 0.2
    norm_obs: bool = True

    archi: ArchiConfig = dataclasses.field(default_factory=ArchiConfig)

class FBModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = config_from_dict(kwargs, Config)

        arch: ArchiConfig = self.cfg.archi

        obs_dim = self.cfg.obs_dim
        obs_history_horizon = self.cfg.obs_history_horizon
        obs_privileges_dim = self.cfg.obs_privileges_dim
        action_dim = self.cfg.action_dim

        critic_obs_dim = obs_dim * (obs_history_horizon + 1) + action_dim * obs_history_horizon + obs_privileges_dim
        backward_obs_dim = obs_dim + obs_privileges_dim
        disc_obs_dim = obs_dim + obs_privileges_dim
        actor_obs_dim = obs_dim * (obs_history_horizon + 1) + action_dim * obs_history_horizon

        # create networks
        self._backward_map_ = build_backward(backward_obs_dim, arch.z_dim, arch.b)
        self._forward_map_ = build_forward(critic_obs_dim, arch.z_dim, action_dim, arch.f)
        self._actor_ = build_actor(actor_obs_dim, arch.z_dim, action_dim, arch.actor)

        self._discriminator_ = build_discriminator(disc_obs_dim, arch.z_dim, arch.discriminator)
        self._critic_ = build_forward(critic_obs_dim, arch.z_dim, action_dim, arch.critic, output_dim=1)
        self._auxi_critic_ = build_forward(critic_obs_dim, arch.z_dim, action_dim, arch.auxi_critic, output_dim=1)

        self._obs_normalizer = nn.BatchNorm1d(obs_dim, affine=False, momentum=0.01) if self.cfg.norm_obs else nn.Identity()
        self._obs_privileges_normalizer = nn.BatchNorm1d(obs_privileges_dim, affine=False, momentum=0.01) if self.cfg.norm_obs else nn.Identity()
        self._action_normalizer = nn.BatchNorm1d(action_dim, affine=False, momentum=0.01) if self.cfg.norm_obs else nn.Identity()

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        self._target_backward_map_ = copy.deepcopy(self._backward_map_)
        self._target_forward_map_ = copy.deepcopy(self._forward_map_)

        self._target_critic_ = copy.deepcopy(self._critic_)
        self._target_auxi_critic_ = copy.deepcopy(self._auxi_critic_)

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.cfg.device = device.type  # type: ignore
        return super().to(*args, **kwargs)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        return load_model(path, device, cls=cls)

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        safetensors_save_model(self, output_folder / "model.safetensors")
        with (output_folder / "config.json").open("w+") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

    def normalize(self, status: dict):
        out = {}
        if "obs" in status:
            obs = status["obs"]
            if obs.dim() == 3:
                _b, _n, _d = obs.shape
                obs = torch.reshape(obs, (_b * _n, _d))
                obs = self._obs_normalizer(obs)
                obs = torch.reshape(obs, (_b, _n, _d))
            else:
                obs = self._obs_normalizer(obs)
            out["obs"] = obs

        if "action" in status:
            action = status["action"]
            if action.dim() == 3:
                _b, _n, _d = action.shape
                action = torch.reshape(action, (_b * _n, _d))
                action = self._action_normalizer(action)
                action = torch.reshape(action, (_b, _n, _d))
            else:
                action = self._action_normalizer(action)
            out["action"] = action

        if "privileges" in status:
            privileges = status["privileges"]
            privileges = self._obs_privileges_normalizer(privileges)
            out["privileges"] = privileges
        return out

    @torch.no_grad()
    def _normalize(self, obs: tuple[torch.Tensor]):
        with eval_mode(self._obs_normalizer), \
            eval_mode(self._action_normalizer), \
            eval_mode(self._obs_privileges_normalizer):

            return self.normalize(obs)

    @torch.no_grad()
    def backward_map(self, obs: dict[torch.Tensor]):
        obs = self._normalize(obs)
        obs = self._build_back_obs(obs)
        return self._backward_map_(obs)

    @torch.no_grad()
    def forward_map(self, obs: dict[torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        obs = self._normalize(obs)
        obs = self._build_critic_obs(obs)
        return self._forward_map_(obs, z, action)

    @torch.no_grad()
    def actor(self, obs: dict[torch.Tensor], z: torch.Tensor, std: float):
        obs = self._normalize(obs)

        return self._actor(obs, z, std)

    @torch.no_grad()
    def critic(self, obs: dict[torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        obs = self._normalize(obs)
        obs = self._build_critic_obs(obs)
        return self._critic_(obs, z, action)

    @torch.no_grad()
    def discriminator(self, obs: dict[torch.Tensor], z: torch.Tensor):
        obs = self._normalize(obs)
        obs = self._build_back_obs(obs)
        return self._discriminator_(obs, z)

    @torch.no_grad()
    def auxi_critic(self, obs: dict[torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        obs = self._normalize(obs)
        obs = self._build_critic_obs(obs)
        return self._auxi_critic_(obs, z, action)


    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(self, obs: dict[torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean
        return dist.sample()

    def reward_inference(self, next_obs: dict[torch.Tensor], reward: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        num_batches = int(np.ceil(next_obs[0].shape[0] / self.cfg.inference_batch_size))
        z = 0
        wr = reward if weight is None else reward * weight
        for i in range(num_batches):
            start_idx, end_idx = i * self.cfg.inference_batch_size, (i + 1) * self.cfg.inference_batch_size

            sample = next_obs[0][start_idx:end_idx].to(self.cfg.device), next_obs[1][start_idx:end_idx].to(self.cfg.device)
            B = self.backward_map(sample)
            z += torch.matmul(wr[start_idx:end_idx].to(self.cfg.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, next_obs: dict[torch.Tensor], reward: torch.Tensor) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: dict[torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def tracking_inference(self, next_obs: dict[torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)

    #
    #
    #
    #
    def _build_back_obs(self, status: dict[torch.Tensor]):
        obs, obs_privileges = status["obs"], status["privileges"]
        obs = obs[:, -1, :]
        obs = torch.cat([obs, obs_privileges], dim = -1)
        return obs

    def _build_critic_obs(self, status: dict[torch.Tensor]):
        obs, action, obs_privileges = status["obs"], status["action"], status["privileges"]
        obs = torch.reshape(obs, (obs.shape[0], -1))
        action = torch.reshape(action, (action.shape[0], -1))
        obs = torch.cat([obs, action, obs_privileges], dim = -1)
        return obs

    def _build_actor_obs(self, status: dict[torch.Tensor]):
        obs, action = status["obs"], status["action"]
        obs = torch.reshape(obs, (obs.shape[0], -1))
        action = torch.reshape(action, (action.shape[0], -1))
        obs = torch.cat([obs, action], dim = -1)
        return obs
    #
    #
    #
    def _actor(self, obs: dict[torch.Tensor], z: torch.Tensor, std: float):
        obs = self._build_actor_obs(obs)
        return self._actor_(obs, z, std)

    def _forward_map(self, obs: dict[torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        obs = self._build_critic_obs(obs)
        return self._forward_map_(obs, z, action)

    def _target_forward_map(self, obs: dict[torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        obs = self._build_critic_obs(obs)
        return self._target_forward_map_(obs, z, action)

    def _backward_map(self, obs: dict[torch.Tensor]):
        obs = self._build_back_obs(obs)
        return self._backward_map_(obs)

    def _target_backward_map(self, obs: dict[torch.Tensor]):
        obs = self._build_back_obs(obs)
        return self._target_backward_map_(obs)

    def _critic(self, obs: dict[torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        obs = self._build_critic_obs(obs)
        return self._critic_(obs, z, action)

    def _target_critic(self, obs: dict[torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        obs = self._build_critic_obs(obs)
        return self._target_critic_(obs, z, action)

    def _discriminator(self, obs: dict[torch.Tensor], z: torch.Tensor):
        obs = self._build_back_obs(obs)
        return self._discriminator_(obs, z)

    def _compute_logits(self, obs: dict[torch.Tensor], z: torch.Tensor):
        obs = self._build_back_obs(obs)
        return self._discriminator_.compute_logits(obs, z)

    def _compute_reward(self, obs: dict[torch.Tensor], z: torch.Tensor):
        obs = self._build_back_obs(obs)
        return self._discriminator_.compute_reward(obs, z)

    def _auxi_critic(self, obs: dict[torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        obs = self._build_critic_obs(obs)
        return self._auxi_critic_(obs, z, action)

    def _target_auxi_critic(self, obs: dict[torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        obs = self._build_critic_obs(obs)
        return self._target_auxi_critic_(obs, z, action)
