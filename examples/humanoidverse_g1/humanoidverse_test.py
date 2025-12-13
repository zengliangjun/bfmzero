import os
import os.path as osp
import sys
root = osp.join(osp.dirname(__file__), "../..")

if root not in sys.path:
    sys.path.insert(0, root)

os.chdir(root)

from humanoidverse_env import motions_main
motions_main.humanoidverse_test()


import torch
torch.set_float32_matmul_precision("high")
from metamotivo.fb_bfmzero import model

from metamotivo.fb_humanoidverse.humanoidverse import envs
from metamotivo.fb_humanoidverse.buffers import motion_buffer
from metamotivo.fb_humanoidverse.humanoidverse import configs
from metamotivo.fb_humanoidverse.collect import history_action_collect

if __name__ == "__main__":
    collect_config: history_action_collect.CollectConfig = history_action_collect.CollectConfig()
    runContext = history_action_collect.RunContext(collect_config)

    env = envs.MotionsWarp()
    num_envs = env.task.num_envs

    motions_config: motion_buffer.MotionBufferConfig = motion_buffer.MotionBufferConfig()
    motions_config.motions_root = configs.motions_root
    buffer = motions_config.make_buffer()

    fbmodel:model.FBModel  = model.FBModel.load("/workspace/data2/VSCODE/MOTION/FBMODULES/FB_2504.11054_CPR_GYM/logs/humanoidverse_bfmzero/run_2025_12_05_14_12_53/checkpoint/model")
    fbmodel.eval()

    runContext.reset(env)


    while True:
        motions = buffer.tracking_motions(num_envs)
        obs, privileges =  motions["observations"], motions["privileges"]

        tracking_zs = []
        for i in range(num_envs):
            obsi = {
                    "obs": obs[i].to(fbmodel.cfg.device),
                    "privileges": privileges[i].to(fbmodel.cfg.device)
                    }
            tracking_z = fbmodel.tracking_inference(obsi)
            tracking_zs.append(tracking_z)
        tracking_zs = torch.stack(tracking_zs, dim = 0)

        for i in range(tracking_zs.shape[1]):
            obs = runContext.observations(fbmodel.cfg.device)
            action = fbmodel.act(obs, tracking_zs[:, i,])
            runContext.step(env, action)


