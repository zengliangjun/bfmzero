import os
import sys
import os.path as osp
root = osp.join(osp.dirname(__file__), "../..")

if root not in sys.path:
    sys.path.insert(0, root)

os.chdir(root)

from humanoidverse_env import main

env = main.humanoidverse_start()
