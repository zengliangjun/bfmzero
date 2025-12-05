import os
import os.path as osp
import sys
root = osp.join(osp.dirname(__file__), "../..")

if root not in sys.path:
    sys.path.insert(0, root)

os.chdir(root)

from humanoidverse_env import motions_main
motions_main.humanoidverse_train()

import dataclasses
import torch
torch.set_float32_matmul_precision("high")
from metamotivo.fb_humanoidverse import workspace
from metamotivo.fb_bfmzero import agent
from metamotivo.fb_humanoidverse.humanoidverse import envs

@dataclasses.dataclass
class TrainConfig(workspace.TrainConfig):
    def __post_init__(self):
        self.name = "humanoidverse_bfmzero"
        super().__post_init__()

@dataclasses.dataclass
class BFMConfig(agent.Config):
    def __post_init__(self):
        self.model.obs_privileges_dim = 448    ## for test

        self.train.lr_f = 3e-4
        self.train.lr_b = 1e-5
        self.train.lr_actor = 3e-4
        self.train.lr_discriminator = 1e-5
        self.train.lr_critic = 3e-4
        self.train.lr_auxi_critic = 3e-4

        self.train.fb_target_tau = 0.01
        self.train.critic_target_tau = 0.005
        self.train.auxi_critic_target_tau = 0.005

        self.train.fb_pessimism_penalty = 0.0
        self.train.actor_pessimism_penalty = 0.5
        self.train.critic_pessimism_penalty = 0.5
        self.train.auxi_critic_pessimism_penalty = 0.5

        self.train.train_goal_ratio = 0.2
        self.train.expert_asm_ratio = 0.6
        self.train.relabel_ratio = 0.8

        self.train.use_mix_rollout = True
        self.train.ortho_coef = 100
        self.train.q_loss_coef = 0.1

        self.train.scale_reg = True
        self.train.reg_coeff = 0.05
        self.train.auxi_reg_coeff = 0.02

        self.train.stddev_clip = 0.3
        self.train.batch_size = 1024
        self.train.discount = 0.98
        self.train.update_z_every_step = 150
        self.train.z_buffer_size = 10000

        self.train.grad_penalty_discriminator = 10.0
        self.train.weight_decay_discriminator = 0.0


if __name__ == "__main__":

    cfg: TrainConfig = TrainConfig()

    agent_cfg: BFMConfig = BFMConfig()
    agent_cfg.model.device = "cuda:0"

    work = workspace.Workspace(cfg, agent_cfg)

    env = envs.MotionsWarp()

    work.train(env)
