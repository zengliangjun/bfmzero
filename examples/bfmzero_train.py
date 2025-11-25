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
        pass

@dataclasses.dataclass
class BFMAgent(agent.BFMAgent):
    def __post_init__(self):
        pass


if __name__ == "__main__":

    cfg: TrainConfig = TrainConfig()
    agent_cfg: BFMAgent = BFMAgent()

    work = workspace.Workspace(cfg, agent_cfg)


    env_cfg: envs.IsaacConfig = envs.IsaacConfig()
    env_cfg.args_cli = args_cli
    env_cfg.workdir = work.work_dir

    env = env_cfg.make_env()

    work.train(env)
