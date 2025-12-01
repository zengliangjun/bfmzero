#!/usr/bin/env python3
"""
关节映射验证工具
用于分析 MuJoCo 关节信息并验证与 npz_play.py 中关节列表的匹配
"""

import sys
import os.path as osp

# MuJoCo 相关导入
root = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
xml_file = osp.join(root, "humanoidverse_env/data/robots/metamotivo_hum/hum.xml")

import mujoco


def get_mujoco_joint_info():
    """获取 MuJoCo 模型中的关节信息"""
    try:
        model = mujoco.MjModel.from_xml_path(xml_file)
        data = mujoco.MjData(model)

        joint_info = []
        print("=== MuJoCo 关节信息 ===")
        print(f"总关节数: {model.njnt}")

        for joint_id in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                joint_range = model.jnt_range[joint_id]
                joint_info.append({
                    'id': joint_id,
                    'name': joint_name,
                    'range': joint_range,
                    'type': model.jnt_type[joint_id]
                })
                print(f"关节 {joint_id}: {joint_name}, 范围: [{joint_range[0]:.3f}, {joint_range[1]:.3f}]")

        return joint_info
    except Exception as e:
        print(f"获取 MuJoCo 关节信息失败: {e}")
        return []


def validate_npz_play_joint_list(mujoco_joints):
    """验证 npz_play.py 中使用的关节列表与 MuJoCo 模型的匹配"""
    print("\n=== 验证 npz_play.py 关节列表 ===")

    # npz_play.py 中使用的关节列表
    npz_play_joints = [
        'L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'L_Knee_x', 'L_Knee_y', 'L_Knee_z',
        'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 'L_Toe_x', 'L_Toe_y', 'L_Toe_z',
        'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z',
        'R_Ankle_x', 'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z',
        'Torso_x', 'Torso_y', 'Torso_z',
        'Spine_x', 'Spine_y', 'Spine_z',
        'Chest_x', 'Chest_y', 'Chest_z',
        'Neck_x', 'Neck_y', 'Neck_z',
        'Head_x', 'Head_y', 'Head_z',
        'L_Thorax_x', 'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z',
        'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z',
        'L_Hand_x', 'L_Hand_y', 'L_Hand_z',
        'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z',
        'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 'R_Wrist_z',
        'R_Hand_x', 'R_Hand_y', 'R_Hand_z'
    ]

    mujoco_joint_names = [joint['name'] for joint in mujoco_joints]

    # 检查 npz_play.py 关节列表的完整性
    missing_in_mujoco = set(npz_play_joints) - set(mujoco_joint_names)

    print(f"npz_play.py 关节列表长度: {len(npz_play_joints)}")
    print(f"MuJoCo 模型关节数量: {len(mujoco_joint_names)}")

    if missing_in_mujoco:
        print(f"在 MuJoCo 中缺失的关节 ({len(missing_in_mujoco)}): {sorted(missing_in_mujoco)}")
    else:
        print("✓ npz_play.py 关节列表在 MuJoCo 模型中完整")

    # 检查额外关节
    extra_in_mujoco = set(mujoco_joint_names) - set(npz_play_joints)

    if extra_in_mujoco:
        print(f"MuJoCo 中额外未使用的关节 ({len(extra_in_mujoco)}): {sorted(extra_in_mujoco)}")

    # 检查关节顺序
    print("\n=== 关节顺序分析 ===")
    mujoco_name_to_id = {joint['name']: joint['id'] for joint in mujoco_joints}

    # 创建 npz_play.py 关节列表的索引映射
    npz_play_order = []
    for joint_name in npz_play_joints:
        if joint_name in mujoco_name_to_id:
            npz_play_order.append((joint_name, mujoco_name_to_id[joint_name]))

    # 按 MuJoCo 索引排序
    npz_play_order_sorted = sorted(npz_play_order, key=lambda x: x[1])

    print("npz_play.py 关节列表在 MuJoCo 中的实际顺序:")
    for i, (joint_name, mujoco_id) in enumerate(npz_play_order_sorted):
        print(f"  {i:2d}: {joint_name} (MuJoCo索引: {mujoco_id})")

    # 检查顺序是否一致
    is_ordered = all(npz_play_order_sorted[i][1] == i for i in range(len(npz_play_order_sorted)))
    if is_ordered:
        print("✓ 关节顺序与 MuJoCo 索引顺序一致")
    else:
        print("⚠ 关节顺序与 MuJoCo 索引顺序不一致，可能需要调整映射")


def main():
    """主函数"""
    print("开始关节映射验证...")

    # 获取 MuJoCo 关节信息
    mujoco_joints = get_mujoco_joint_info()
    if not mujoco_joints:
        print("无法获取 MuJoCo 关节信息，退出验证")
        return

    # 验证 npz_play.py 关节列表
    validate_npz_play_joint_list(mujoco_joints)

    # 总结
    print("\n=== 验证总结 ===")
    print("此验证仅检查 MuJoCo 模型与 npz_play.py 关节列表的匹配")
    print("由于 Isaac Sim 环境依赖问题，Isaac Sim 部分的验证需要在实际运行环境中进行")


if __name__ == "__main__":
    main()
