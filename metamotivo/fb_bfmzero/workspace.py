
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
from metamotivo.fb_bfmzero.buffers import isaac_buffer

@dataclasses.dataclass
class TrainConfig:
    name: str = "fb_bfmzero"
    seed: int = 0
    motions_buffer: isaac_buffer.MotionBufferConfig = isaac_buffer.MotionBufferConfig()

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

    @torch.no_grad()
    def _train_collect_one_step(self, env, context_items):
        step = context_items['step']
        context_z = context_items['context_z']
        done = context_items['done']
        observations = context_items['observations']
        privileges = context_items['privileges']

        step_count = env.timestep()[:, None].to(self.agent.device)
        context_z = self.agent.maybe_update_rollout_context(z=context_z, step_count=step_count)

        if step < self.cfg.seed_steps:
            action = env.sample_action()
        else:
            # this works in inference mode
            observations = observations.to(self.agent.device)
            privileges = privileges.to(self.agent.device)

            action = self.agent.act(obs=(observations, privileges), z=context_z, mean=False)

        next_obs, rewards, next_dones, next_infos = env.step(action)

        ## update context
        context_items['context_z'] = context_z
        context_items['done'] = next_dones
        context_items['observations'] = next_infos['observations']
        context_items['privileges'] = next_infos['privileges']

        if done.shape[0] == torch.sum(done.float()):
            return None

        indexes = ~done
        data = {
            "observations": observations[indexes],
            "privileges": privileges[indexes],
            "action": action[indexes],
            "z": context_z[indexes],
            "step_count": step_count[indexes],
            "next": {
                "observations": next_infos['observations'][indexes],
                "privileges": next_infos['privileges'][indexes],
                "rewards": rewards[indexes][:, None],
                "terminated": next_infos["terminated"][indexes][:, None],
                "truncated": next_infos["truncated"][indexes][:, None]
            },
        }
        return data


    def train(self, env):
        self.train_online(env)

    def train_online(self, env) -> None:
        replay_buffer = {
            "train": DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device),
            "expert_slicer": self.cfg.motions_buffer.make_buffer(),
        }


        obs, extras = env.reset()
        done = torch.zeros(self.agent_cfg.train.batch_size, dtype=torch.bool)

        collect_context = {
                "observations": extras["observations"],
                "privileges": extras["privileges"],
                'done': done,
                "context_z": None
            }

        start_time = time.time()
        fps_start_time = time.time()

        total_metrics = None

        progb = tqdm(total=self.cfg.max_steps)
        for step in range(1, self.cfg.max_steps + 1):

            for _ in range(self.cfg.one_step_collect_iters):
                collect_context['step'] = step
                data = self._train_collect_one_step(env, collect_context)
                if data is not None:
                    replay_buffer["train"].extend(data)

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

            progb.update(1)

        self.agent.save(osp.join(self.work_dir, "checkpoint"))
