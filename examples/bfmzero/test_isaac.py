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

args_cli.headless = False
args_cli.task = "FB-G129dof-Play-v0"

# launch omniverse app
app_launcher = AppLauncher(args_cli)



import os
import sys
import os.path as osp

script_dir = osp.abspath(osp.dirname(__file__))
workroot_dir = osp.abspath(osp.join(script_dir, "../.."))
os.chdir(workroot_dir)

source_dir = f"{workroot_dir}"
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

import metamotivo.fb_bfmzero.isaac

from metamotivo.fb_bfmzero.isaac import envs

if __name__ == "__main__":

    cfg: envs.IsaacConfig  = envs.IsaacConfig()
    cfg.args_cli = args_cli
    cfg.workdir = f"{workroot_dir}/logs"

    env = cfg.make_env()

    obs, extras = env.reset()
    action = envs.sample_action()
    objs, obs, rewards, dones, extras = envs.step(action)

