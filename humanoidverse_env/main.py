
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

import logging
from loguru import logger
import sys
import os
import os.path as osp

if False:
    HumanoidVerse_root = "/workspace/data2/VSCODE/humanoid_motions/WORKSPACE_FORME/HumanoidVerse"
    if HumanoidVerse_root not in sys.path:
        sys.path.insert(0, HumanoidVerse_root)

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from humanoidverse_env import hydra_main

simulator_type = None
simulation_app = None
humanoidverse_task = None

@hydra_main.main(config_path="configs", config_name="base", version_base="1.1")
def humanoidverse_start(config: OmegaConf):
    # import ipdb; ipdb.set_trace()
    global humanoidverse_task
    global simulation_app
    simulator_type = config.simulator['_target_'].split('.')[-1]
    # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="bfmzero")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless

        app_launcher = AppLauncher(args_cli)
        global simulation_app
        simulation_app = app_launcher.app

        # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacGym':
        import isaacgym  # noqa: F401

    # have to import torch after isaacgym
    import torch  # noqa: E402
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge

    # logging to hydra log file
    hydra_log_path = osp.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pre_process_config(config)

    config.env.config.save_rendering_dir = osp.join(config.experiment_dir, "renderings_training")
    env: BaseTask = instantiate(config=config.env, device=device)
    humanoidverse_task = env

def humanoidverse_final():
    global simulator_type, simulation_app
    if simulator_type == 'IsaacSim':
        simulation_app.close()
