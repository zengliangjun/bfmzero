
import os
import sys
import os.path as osp
import argparse
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_dir", type=str, required=False, help="The path to the input motion csv file.")
parser.add_argument("--out_filename", type=str, required=False, help="The path to the input motion csv file.")

args_cli = parser.parse_args()

args_cli.input_dir = '/workspace.data1/ISAACSIM45/MATA/data/SPLITDATA/motions/g1/LAFAN1_Retargeting_Dataset' 
args_cli.out_filename = "/workspace.data1/ISAACSIM45/MATA/data/SPLITDATA/motions/g1/g1_50_LAFAN1"

def work_dir():

    full_items = {}
    for root, dirs, files in os.walk(args_cli.input_dir):
        for file in files:
            # print(file)
            if not file.endswith(".npz") and not file.endswith(".npy"):
                continue
            full_file = osp.join(root, file)
            name = full_file.replace(args_cli.input_dir, "")

            out_items = {}
            with open(full_file, 'r+b') as fd:
                items = np.load(fd)

                for file in items.files:
                    out_items[file] = items[file]

            full_items[name] = out_items

    np.savez(args_cli.out_filename, **full_items)


if __name__ == "__main__":
    work_dir()

