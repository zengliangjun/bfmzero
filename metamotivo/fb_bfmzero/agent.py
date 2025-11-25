
import dataclasses
from typing import Dict

import torch
import torch.nn.functional as F
from torch import autograd

from ..fb_cpr.agent import FBcprAgent, TrainConfig as FBTrainConfig
from ..nn_models import weight_init, _soft_update_params, eval_mode
from ..misc.zbuffer import ZBuffer

from .model import Config as ModelConfig
from .model import Model, config_from_dict


@dataclasses.dataclass
class TrainConfig(FBTrainConfig):
    lr_auxi_critic: float = 3e-4
    auxi_critic_target_tau: float = 0.005
    auxi_critic_pessimism_penalty: float = 0.5
    auxi_reg_coeff: float = 1


@dataclasses.dataclass
class Config:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    cudagraphs: bool = False
    compile: bool = False


class BFMAgent(FBcprAgent):
    def __init__(self, **kwargs):
        # make sure batch size is a multiple of seq_length
        seq_length = kwargs["model"]["seq_length"]
        batch_size = kwargs["train"]["batch_size"]
        kwargs["train"]["batch_size"] = int(torch.ceil(torch.tensor([batch_size / seq_length])) * seq_length)
        del seq_length, batch_size

        self.cfg = config_from_dict(kwargs, Config)
        self._model = Model(**dataclasses.asdict(self.cfg.model))
        self._model.to(self.cfg.model.device)
        self.setup_training()
        self.setup_compile()

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()
        self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self.cfg.model.device)

        ## fb
        self.backward_optimizer = torch.optim.Adam(
            self._model._backward_map_.parameters(),
            lr=self.cfg.train.lr_b,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            self._model._forward_map_.parameters(),
            lr=self.cfg.train.lr_f,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor_.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

        # prepare parameter list
        self._forward_map_paramlist = tuple(x for x in self._model._forward_map_.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map_.parameters())
        self._backward_map_paramlist = tuple(x for x in self._model._backward_map_.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map_.parameters())

        # fb cpr
        self.critic_optimizer = torch.optim.Adam(
            self._model._critic_.parameters(),
            lr=self.cfg.train.lr_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator_.parameters(),
            lr=self.cfg.train.lr_discriminator,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay_discriminator,
        )

        self._critic_map_paramlist = tuple(x for x in self._model._critic_.parameters())
        self._target_critic_map_paramlist = tuple(x for x in self._model._target_critic_.parameters())

        # bfm zero
        self._auxi_critic_paramlist = tuple(x for x in self._model._auxi_critic.parameters())
        self._target_auxi_critic_paramlist = tuple(x for x in self._model._target_auxi_critic.parameters())

        self.critic_optimizer = torch.optim.Adam(
            self._model._auxi_critic.parameters(),
            lr=self.cfg.train.lr_auxi_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )


    @torch.no_grad()
    def sample_mixed_z(self, train_goal: tuple[torch.Tensor], expert_encodings: torch.Tensor, *args, **kwargs):
        z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
        p_goal = self.cfg.train.train_goal_ratio
        p_expert_asm = self.cfg.train.expert_asm_ratio
        prob = torch.tensor(
            [p_goal, p_expert_asm, 1 - p_goal - p_expert_asm],
            dtype=torch.float32,
            device=self.device,
        )
        mix_idxs = torch.multinomial(prob, num_samples=self.cfg.train.batch_size, replacement=True).reshape(-1, 1)

        # zs obtained by encoding train goals
        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        ################
        train_goal = train_goal[0][perm], train_goal[1][perm]
        goals = self._model._backward_map(train_goal)

        goals = self._model.project_z(goals)
        z = torch.where(mix_idxs == 0, goals, z)

        # zs obtained by encoding expert trajectories
        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

        return z

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs, train_privileges, train_action, train_z, \
        train_next_obs, train_next_privileges, train_terminated, \
            train_rewards = (
            train_batch["observation"].to(self.device),
            train_batch["privileges"].to(self.device),
            train_batch["action"].to(self.device),
            train_batch["z"].to(self.device),

            train_batch["next"]["observation"].to(self.device),
            train_batch["next"]["privileges"].to(self.device),
            train_batch["next"]["terminated"].to(self.device),
            train_batch["next"]["rewards"].to(self.device),
        )
        discount = self.cfg.train.discount * ~train_terminated

        expert_obs, expert_privileges, \
        expert_next_obs, expert_next_privileges = (
            expert_batch["observation"].to(self.device),
            expert_batch["privileges"].to(self.device),
            expert_batch["next"]["observation"].to(self.device),
            expert_batch["next"]["privileges"].to(self.device),
        )

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)

        self._model._obs_privileges_normalizer(train_privileges)
        self._model._obs_privileges_normalizer(train_next_privileges)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer), eval_mode(self._model._obs_privileges_normalizer):
            train_obs, train_next_obs = (
                self._model._obs_normalizer(train_obs),
                self._model._obs_normalizer(train_next_obs),
            )
            train_privileges, train_next_privileges = (
                self._model._obs_privileges_normalizer(train_privileges),
                self._model._obs_privileges_normalizer(train_next_privileges),
            )

            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

            expert_privileges, expert_next_privileges = (
                self._model._obs_privileges_normalizer(expert_privileges),
                self._model._obs_privileges_normalizer(expert_next_privileges),
            )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=[expert_next_obs, expert_next_privileges])


        # train the discriminator
        grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        metrics = self.update_discriminator(
            expert_obs=[expert_obs, expert_privileges],
            expert_z=expert_z,
            train_obs=[train_obs, train_privileges],
            train_z=train_z,
            grad_penalty=grad_penalty,
        )

        metrics.update(
                self.update_auxi_critic(
                    obs=[train_obs, train_privileges],
                    action=train_action,
                    discount=discount,
                    next_obs=[train_next_obs, train_next_privileges],
                    z=train_z,
                    reward=train_rewards
            )
        )

        z = self.sample_mixed_z(train_goal=[train_next_obs, train_next_privileges], expert_encodings=expert_z).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=[train_obs, train_privileges],
                action=train_action,
                discount=discount,
                next_obs=[train_next_obs, train_next_privileges],
                goal=[train_next_obs, train_next_privileges],
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics.update(
            self.update_critic(
                obs=[train_obs, train_privileges],
                action=train_action,
                discount=discount,
                next_obs=[train_next_obs, train_next_privileges],
                z=train_z,
            )
        )
        metrics.update(
            self.update_actor(
                obs=[train_obs, train_privileges],
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._critic_map_paramlist,
                self._target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )
            _soft_update_params(
                self._auxi_critic_map_paramlist,
                self._target_auxi_critic_map_paramlist,
                self.cfg.train.auxi_critic_target_tau,
            )

        return metrics

    def update_auxi_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        z: torch.Tensor,
        reward: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        num_parallel = self.cfg.model.archi.critic.num_parallel
        # compute target critic
        with torch.no_grad():
            dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)

            next_Qs = self._model._target_auxi_critic(next_obs, z, next_action)  # num_parallel x batch x 1
            Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.auxi_critic_pessimism_penalty)
            target_Q = reward + discount * next_V
            expanded_targets = target_Q.expand(num_parallel, -1, -1)

        # compute critic loss
        Qs = self._model._auxi_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
        critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "auxi/target_Q": target_Q.mean().detach(),
                "auxi/reward": reward.mean().detach(),
                "auxi/auxi_Q": Qs.mean().detach(),
                "auxi/mean_next_Q": Q_mean.mean().detach(),
                "auxi/unc_Q": Q_unc.mean().detach(),
                "auxi/loss": critic_loss.mean().detach(),
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        dist = self._model._actor(obs, z, self._model.cfg.actor_std)
        action = dist.sample(clip=self.cfg.train.stddev_clip)

        # compute discriminator reward loss
        Qs_auxi = self._model._auxi_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
        _, _, Qs_auxi = self.get_targets_uncertainty(Qs_auxi, self.cfg.train.actor_pessimism_penalty)  # batch

        # compute discriminator reward loss
        Qs_discriminator = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
        _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.cfg.train.actor_pessimism_penalty)  # batch

        # compute fb reward loss
        Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
        Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

        weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
        actor_loss = -(Q_discriminator * self.cfg.train.reg_coeff + Qs_auxi * self.cfg.train.auxi_reg_coeff) * weight - Q_fb
        actor_loss = actor_loss.mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "actor/loss": actor_loss.detach(),
                "actor/Q_dis": Q_discriminator.mean().detach(),
                "actor/Q_fb": Q_fb.mean().detach(),
                "actor/Q_auxi": Qs_auxi.mean().detach(),
            }
        return output_metrics
