import numpy as np
import pickle

file = "/workspace.data1/ISAACSIM45/MATA/data/SPLITDATA/PHUMA/data/g1/custom/walk_chunk_0050.npy"


dof_pos = data['dof_pos']
root_trans = data['root_trans']
root_ori = data['root_ori']
fps = data['fps']

print(dof_pos.shape)
print(root_trans.shape)
print(root_ori.shape)
