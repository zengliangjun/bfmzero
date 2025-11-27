import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")

parser.add_argument("--track_robot", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--debugs", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--duration", type=int, default=2, help="Run in real-time, if possible.")


parser.add_argument("--proprioceptive-history-length", type=int, default=None, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

args_cli.headless = True
args_cli.task = "FB-G129dof-v0"

# launch omniverse app
app_launcher = AppLauncher(args_cli)

import os
import os.path as osp
import sys
work_root = osp.dirname(osp.dirname(__file__))
sys.path.append(work_root)
os.chdir(work_root)


import dataclasses
import torch
torch.set_float32_matmul_precision("high")

import metamotivo.fb_bfmzero.isaac
from metamotivo.fb_bfmzero.isaac import envs
from metamotivo.fb_bfmzero import workspace
from metamotivo.fb_bfmzero import agent


@dataclasses.dataclass
class TrainConfig(workspace.TrainConfig):
    def __post_init__(self):
        self.name = "isaac_bfmzero"
        self.motions_buffer.motions_root = "/workspace/data2/VSCODE/MOTION/MATA/data/SPLITDATA/"

        self.compile = False
        self.cudagraphs = False

@dataclasses.dataclass
class BFMConfig(agent.Config):
    def __post_init__(self):

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
    agent_cfg.model.device = cfg.device

    if args_cli.proprioceptive_history_length is not None:
        agent_cfg.model.obs_history_horizon = args_cli.proprioceptive_history_length
        cfg.motions_buffer.history_horizon = args_cli.proprioceptive_history_length

    work = workspace.Workspace(cfg, agent_cfg)

    env_cfg: envs.IsaacConfig = envs.IsaacConfig()
    env_cfg.args_cli = args_cli
    env_cfg.workdir = work.work_dir

    env = env_cfg.make_env()

    work.train(env)
