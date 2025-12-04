
import sys
import os.path as osp
root = osp.abspath(osp.join(osp.dirname(__file__), "../.."))

xml_file = osp.join(root, "humanoidverse_env/data/robots/metamotivo_hum/hum.xml")

import mujoco
import mujoco_viewer
import json
import os
import time

def init_mujoco():
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    return model, data, viewer

def mujoco_joint_name(model):
    joint_names = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_names.append(joint_name)

    print("mujoco joint name")
    print("+" * 80)
    print(joint_names)
    print("+" * 80)

def mujoco_joint_name2(model, dof_names):
    joint_ids = []
    for dof_name in dof_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof_name)
        joint_ids.append(joint_id)
    print("mujoco joint name2")
    print("+" * 80)
    print("joint_ids:", joint_ids)
    #  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 34, 35, 36, 37, 38, 39, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]

def mujoco_body_name(model):
    body_names = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        body_names.append(body_name)

    print("mujoco_body_name")
    print("+" * 80)
    print(body_names)

def mujoco_body_name2(model, body_names):
    print("mujoco_body_name2")
    print("+" * 80)

    body_ids = []
    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_ids.append(body_id)
    print("body_ids:", body_ids)
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 13, 14, 20, 21, 22, 23, 24


def mujoco_isaac_pos_limit(model):
    print("isaac_pos_limit")
    print("+" * 80)
    lower_limit = []
    upper_limit = []
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_range = model.jnt_range[i]
        is_limited = model.jnt_limited[i]

        if is_limited:
            lower_limit.append(joint_range[0])
            upper_limit.append(joint_range[1])

            print(f"'{joint_name}': [{joint_range[0]:.3f}, {joint_range[1]:.3f}]")
        else:
            print(f"'{joint_name}': no limited")

def mujoco_isaac_pos_limit2(model, dof_names):
    print("isaac_pos_limit2")
    print("+" * 80)
    lower_limit = ""
    upper_limit = ""

    for name in dof_names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_range = model.jnt_range[i]
        is_limited = model.jnt_limited[i]

        if is_limited:
            lower_limit += f"{joint_range[0]:.3f}, "
            upper_limit += f"{joint_range[1]:.3f}, "


    print("lower_limit", lower_limit)
    print("upper_limit", upper_limit)


def mujoco_isaac_effort_limit(model):
    print("isaac_effort_limit")
    print("+" * 80)
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        force_range = model.actuator_forcerange[i]
        print(f"'{actuator_name}': {force_range[1]:.3f}")

def mujoco_isaac_effort_limit2(model, dof_names):
    print("isaac_effort_limit2")
    print("+" * 80)
    limit = ""
    for name in dof_names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        force_range = model.actuator_forcerange[i]
        limit += f"{force_range[1]:.3f}, "

    print("limit", limit)

def mujoco_isaac_stiffness(model):
    print("isaac_stiffness")
    print("+" * 80)
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        stiffness = model.jnt_stiffness[i]
        print(f"'{joint_name}': {stiffness}")


def mujoco_isaac_damping(model):
    print("isaac_damping")
    print("+" * 80)
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        damping = model.dof_damping[i]
        print(f"'{joint_name}': {damping}")


if __name__ == "__main__":
    model, data, viewer = init_mujoco()
    mujoco_isaac_pos_limit(model)
    mujoco_isaac_effort_limit(model)
    #mujoco_isaac_stiffness(model)
    #mujoco_isaac_damping(model)
    dof_names = ['L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'L_Knee_x', 'L_Knee_y', 'L_Knee_z', \
              'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 'L_Toe_x', 'L_Toe_y', 'L_Toe_z', \
              'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z', \
              'R_Ankle_x', 'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z', \
              'Torso_x', 'Torso_y', 'Torso_z', \
              'Spine_x', 'Spine_y', 'Spine_z', \
              'Chest_x', 'Chest_y', 'Chest_z', \
              'L_Thorax_x', 'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', \
              'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', \
              'L_Hand_x', 'L_Hand_y', 'L_Hand_z', \
              'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z', \
              'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', \
              'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 'R_Wrist_z', \
              'R_Hand_x', 'R_Hand_y', 'R_Hand_z']


    body_names = ['Pelvis',
               'L_Hip', 'L_Knee',
               'L_Ankle','L_Toe',
               'R_Hip','R_Knee',
               'R_Ankle','R_Toe',
               'Torso',
               'Spine',
               'Chest',
               'L_Thorax', 'L_Shoulder',
               'L_Elbow', 'L_Wrist',
               'L_Hand',
               'Neck', 'Head',
               'R_Thorax','R_Shoulder',
               'R_Elbow', 'R_Wrist',
               'R_Hand']
    mujoco_joint_name2(model, dof_names)
    mujoco_body_name2(model, body_names)

    mujoco_isaac_pos_limit2(model, dof_names)
    mujoco_isaac_effort_limit2(model, dof_names)



