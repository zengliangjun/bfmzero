import os
import torch
import os.path as osp
import argparse
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_dir", type=str, required=False, help="The path to the input motion csv file.")
parser.add_argument("--out_filename", type=str, required=False, help="The path to the input motion csv file.")

args_cli = parser.parse_args()

args_cli.input_dir = '/workspace/data2/VSCODE/MOTION/FBMODULES/data/motions/g1/LAFAN1_Retargeting_Dataset/g1'
args_cli.out_filename = "/workspace/data2/VSCODE/MOTION/FBMODULES/data/motions/g1/mergers/g1_LAFAN1_Retargeting_Dataset.pth"

def work_dir():

    full_items = {}
    for root, dirs, files in os.walk(args_cli.input_dir):
        for file in files:
            # print(file)
            if not file.endswith(".pth") and not file.endswith(".pth"):
                continue
            full_file = osp.join(root, file)
            name = full_file.replace(args_cli.input_dir, "")

            full_items[name] = torch.load(full_file)

    torch.save(full_items, args_cli.out_filename)


if __name__ == "__main__":
    work_dir()
