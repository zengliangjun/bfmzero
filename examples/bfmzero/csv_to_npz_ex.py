"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python examples/bfmzero/csv_to_npz_ex.py --input_file /workspace.data1/ISAACSIM45/MATA/data/SPLITDATA/csv/g1/LAFAN1_Retargeting_Dataset --input_fps 30 \
    --output_name /workspace.data1/ISAACSIM45/MATA/data/SPLITDATA/motions/g1/LAFAN1_Retargeting_Dataset/ --output_fps 50 --headless


"""

import csv_to_npz


import os
import sys
import os.path as osp
root0 = osp.join(osp.dirname(__file__), "../..")

source_dir = f"{root0}/sources"
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=csv_to_npz.args_cli.device)
    sim_cfg.dt = 1.0 / csv_to_npz.args_cli.output_fps
    sim = sim_utils.SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = csv_to_npz.ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = csv_to_npz.InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator

    if not osp.isdir(csv_to_npz.args_cli.input_file):
        base_dir = osp.dirname(csv_to_npz.args_cli.input_file)
        inputfiles = [csv_to_npz.args_cli.input_file]
    else:
        base_dir = csv_to_npz.args_cli.input_file

        inputfiles = []
        for root, dirs, files in os.walk(csv_to_npz.args_cli.input_file):
            if -1 == root.find("g1"):
                continue

            for file in files:
                if not file.endswith(".csv"):
                    continue
                full_file = osp.join(root, file)
                inputfiles.append(full_file)

    out_dir = csv_to_npz.args_cli.output_name

    for inputfile in inputfiles:

        csv_to_npz.args_cli.input_file = inputfile
        output_name = inputfile.replace(base_dir, out_dir)
        output_name = output_name.replace(".csv", "")
        csv_to_npz.args_cli.output_name = output_name
        csv_to_npz.root0 = root0

        csv_to_npz.run_simulator(
            sim,
            scene,
            joint_names=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    csv_to_npz.simulation_app.close()

