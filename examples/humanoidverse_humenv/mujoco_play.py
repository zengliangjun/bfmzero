
import sys
import os.path as osp
import time
import numpy as np

root = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
xml_file = osp.join(root, "humanoidverse_env/data/robots/metamotivo_hum/hum.xml")

import mujoco
import mujoco_viewer
import h5py
import humenv

def init_mujoco():
    """初始化 MuJoCo 环境"""
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    return model, data, viewer

# 配置参数
input_file = "/workspace/ISAACSIM45ENVS/META/humenv/data_preparation/humenv_amass/0-ACCAD_Female1General_c3d_A2-Sway_poses.hdf5"
input_fps = 30
loop_playback = True  # 是否循环播放

def _load_motion(motion_file, keys=["qpos", "qvel"]):
    """加载运动数据"""
    try:
        hf = h5py.File(motion_file, "r")
        num_ep = hf.attrs["num_episodes"]
        if num_ep == 0:
            raise ValueError("No episodes found in the motion file")

        # 加载第一个episode的数据
        episode = hf[f"ep_0"]
        ep = {k: episode[k][:] for k in episode.keys()}

        print(f"成功加载运动数据: {motion_file}")
        print(f"数据形状 - qpos: {ep['qpos'].shape}, qvel: {ep['qvel'].shape}")
        print(f"数据范围 - qpos: [{ep['qpos'].min():.3f}, {ep['qpos'].max():.3f}], "
              f"qvel: [{ep['qvel'].min():.3f}, {ep['qvel'].max():.3f}]")

        hf.close()
        return ep
    except Exception as e:
        print(f"加载运动数据失败: {e}")
        return None

def play_data(model, data, viewer, input_file, fps=30, loop=True):
    """播放运动数据"""
    episode = _load_motion(input_file)
    if episode is None:
        print("无法加载运动数据，退出播放")
        return

    qpos = episode["qpos"]
    qvel = episode["qvel"]
    observations = episode['observation']

    total_frames = qpos.shape[0]
    frame_duration = 1.0 / fps

    print(f"开始播放运动数据: {total_frames} 帧, FPS: {fps}")
    print("按 ESC 键退出播放")

    frame_id = 0
    start_time = time.time()

    while viewer.is_alive:
        # 设置当前帧的状态
        data.qpos[:] = qpos[frame_id]
        data.qvel[:] = qvel[frame_id]

        # 前向动力学计算
        mujoco.mj_forward(model, data)

        if False:
            # 渲染当前帧
            viewer.render()

            # 计算下一帧的时间
            elapsed_time = time.time() - start_time
            target_time = (frame_id + 1) * frame_duration

            # 帧率控制
            if elapsed_time < target_time:
                time.sleep(target_time - elapsed_time)
        else:
            mujoco.mj_kinematics(model, data)
            obs_dict = humenv.env.compute_humanoid_self_obs_v2(
                model,
                data,
                upright_start=False,
                root_height_obs=True,
                humanoid_type="smpl",
            )

            observation = np.concatenate([v.ravel() for v in obs_dict.values()], axis=0, dtype=np.float64)
            diff = observations[frame_id] - observation
            print(diff)

        # 更新帧索引
        frame_id += 1

        # 显示进度
        if frame_id % 30 == 0:  # 每30帧显示一次进度
            progress = (frame_id / total_frames) * 100
            print(f"播放进度: {frame_id}/{total_frames} ({progress:.1f}%)")

        # 检查是否到达末尾
        if frame_id >= total_frames:
            if loop:
                print("循环播放...")
                frame_id = 0
                start_time = time.time()  # 重置计时器
            else:
                print("播放完成")
                break

    print("播放结束")

if __name__ == "__main__":
    try:
        model, data, viewer = init_mujoco()

        joint_names = []
        for joint_id in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            joint_names.append(joint_name)

        print("=" * 80)
        print(joint_names)
        print("=" * 80)

        print("MuJoCo 环境初始化成功")

        play_data(model, data, viewer, input_file, fps=input_fps, loop=loop_playback)

    except Exception as e:
        print(f"运行错误: {e}")
    finally:

        if 'viewer' in locals():
            viewer.close()
        print("程序退出")
