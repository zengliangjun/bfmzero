
import dataclasses
import json
import random
import time
import numpy as np

from tqdm import tqdm
from datetime import datetime
import os
import os.path as osp
from loguru import logger as ulogger
import torch


from metamotivo.buffers.buffers import DictBuffer
from metamotivo.fb_bfmzero import agent
from metamotivo.fb_bfmzero.logger import logger
from metamotivo.fb_bfmzero.buffers import motion_buffer, replay_buffer as replay

from metamotivo.fb_bfmzero.collect import history_action_collect
from metamotivo.fb_bfmzero.isaac import configs

@dataclasses.dataclass
class TrainConfig:
    name: str = "fb_bfmzero"
    seed: int = 0
    motions_buffer: motion_buffer.MotionBufferConfig = motion_buffer.MotionBufferConfig()
    replay_buffer: replay.ReplayBuffer = replay.ReplayBuffer(capacity = 0)
    collect_buffer: history_action_collect.CollectConfig = history_action_collect.CollectConfig()

    buffer_size: int = 5_000_000

    # steps
    max_steps: int = 192_000
    seed_steps: int = 10
    one_step_collect_iters: int = 1
    one_step_update_iters: int = 16

    log_every_steps: int = 1 #2_000
    checkpoint_every_steps: int = 2_000
    eval_every_steps: int = 2_000

    # work dir
    work_dir: str | None = None

    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2

    # misc
    compile: bool = False
    cudagraphs: bool = False
    device: str = "cuda"
    buffer_device: str = "cpu"

    # eval
    evaluate: bool = False
    eval_every_steps: int = 1_000_000


    def __post_init__(self):
        self.motions_buffer.motions_root = configs.motions_root
        self.motions_buffer.observations_key = ["base_ang_vel", "gravity", "joint_pos" ,"joint_vel"]
        self.motions_buffer.privileges_key = [
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

        self.collect_buffer.observations_key = self.motions_buffer.observations_key
        self.collect_buffer.privileges_key = self.motions_buffer.privileges_key
        self.collect_buffer.history_horizon = self.motions_buffer.history_horizon
        self.collect_buffer.action_key = "history_action"

        self.replay_buffer.observations_key = self.collect_buffer.observations_key
        self.replay_buffer.privileges_key = self.collect_buffer.privileges_key
        self.replay_buffer.action_key = self.collect_buffer.action_key

        self.replay_buffer.capacity=self.buffer_size 
        self.replay_buffer.device=self.buffer_device


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class Workspace:

    def _preinit_workdir(self):
        date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        workdir = osp.join(os.getcwd(), "logs", self.cfg.name, f"run_{date}")
        os.makedirs(workdir)
        self.work_dir = workdir
        self.cfg.work_dir = workdir

        ulogger.info(f"init_workdir: {workdir}.")

        self.logger = logger.Logger(workdir,
                             use_tb=True,
                             use_wandb=False,
                             use_hiplog=False)


    def __init__(self, cfg: TrainConfig, agent_cfg: agent.Config) -> None:
        self.cfg = cfg
        self.agent_cfg = agent_cfg

        self._preinit_workdir()

        set_seed_everywhere(self.cfg.seed)
        self.agent =  agent.BFMAgent(**dataclasses.asdict(agent_cfg))

        with open(osp.join(self.work_dir, "train_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

        with open(osp.join(self.work_dir, "agent_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.agent_cfg), f, indent=4)

    def train(self, env):
        self.train_online(env)

    def train_online(self, env) -> None:
        replay_buffer = {
            "train": self.cfg.replay_buffer,
            "expert_slicer": self.cfg.motions_buffer.make_buffer(),
        }

        collect = history_action_collect.CollectContext(self.cfg.collect_buffer)
        collect.reset(env)

        start_time = time.time()
        fps_start_time = time.time()

        total_metrics = None

        progb = tqdm(total=self.cfg.max_steps)
        for step in range(1, self.cfg.max_steps + 1):

            for _ in range(self.cfg.one_step_collect_iters):
                collect.collect_step(step, env, self.agent, replay_buffer["train"])

            if step <= self.cfg.seed_steps:
                continue

            for _ in range(self.cfg.one_step_update_iters):
                metrics = self.agent.update(replay_buffer, step)
                if total_metrics is None:
                    num_metrics_updates = 1
                    total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
                else:
                    num_metrics_updates += 1
                    total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if step % self.cfg.log_every_steps == 0 and total_metrics is not None:
                logger_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / num_metrics_updates
                    logger_dict[k] = np.round(tmp.mean().item(), 6)


                logger_pad = 35
                ulogger_buffer = f"\n{' step:':>{logger_pad}} {step}\n"
                for k in sorted(list(logger_dict.keys())):
                    ulogger_buffer += f"{k:>{logger_pad}}: {logger_dict[k]}\n"


                time_dict = {
                    "duration [minutes]": (time.time() - start_time) / 60,
                    "FPS": (1 if step == 0 else self.cfg.log_every_steps) / (time.time() - fps_start_time)
                }

                for k in sorted(list(time_dict.keys())):
                    ulogger_buffer += f"{k:>{logger_pad}}: {time_dict[k]}\n"

                ulogger.info(ulogger_buffer)

                self.logger.log_metrics(logger_dict, step, ty='train')

                total_metrics = None
                fps_start_time = time.time()



            if step % self.cfg.checkpoint_every_steps == 0:
                self.agent.save(osp.join(self.work_dir, "checkpoint"))

            if step % (self.cfg.checkpoint_every_steps * 10) == 0:
                replay_buffer["train"].save(osp.join(self.work_dir, "checkpoint", "replay_buffer.pth"))

            progb.update(1)

        self.agent.save(osp.join(self.work_dir, "checkpoint"))
        replay_buffer["train"].save(osp.join(self.work_dir, "checkpoint", "replay_buffer.pth"))
