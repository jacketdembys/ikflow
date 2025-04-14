import torch
import numpy as np

# Training D (end-effector poses)
filename = '/home/jacket/.cache/ikflow/datasets/panda/endpoints_tr.pt__tag0=non-self-colliding'
endpoints_tr = torch.load(filename, weights_only=False).cpu()
print(f"Training D:\t{endpoints_tr.shape}")
np.save('/home/jacket/Documents/ikflow/datasets/panda/endpoints_tr.npy', endpoints_tr.numpy())

# Training Q (joint configurations)
filename = '/home/jacket/.cache/ikflow/datasets/panda/samples_tr.pt__tag0=non-self-colliding'
samples_tr = torch.load(filename, weights_only=False).cpu()
print(f"Training Q:\t{samples_tr.shape}")
np.save('/home/jacket/Documents/ikflow/datasets/panda/samples_tr.npy', samples_tr.numpy())

# Testing D (end-effector poses)
filename = '/home/jacket/.cache/ikflow/datasets/panda/endpoints_te.pt__tag0=non-self-colliding'
endpoints_te = torch.load(filename, weights_only=False).cpu()
print(f"Testing D:\t{endpoints_te.shape}")
np.save('/home/jacket/Documents/ikflow/datasets/panda/endpoints_te.npy', endpoints_te.numpy())

# Testing Q (joint configurations)
filename = '/home/jacket/.cache/ikflow/datasets/panda/samples_te.pt__tag0=non-self-colliding'
samples_te = torch.load(filename, weights_only=False).cpu()
print(f"Testing Q:\t{samples_te.shape}")
np.save('/home/jacket/Documents/ikflow/datasets/panda/samples_te.npy', samples_te.numpy())
