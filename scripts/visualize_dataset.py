import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting


filename = '/home/jacket/.cache/ikflow/datasets/panda/endpoints_tr.pt__tag0=non-self-colliding'
endpoints_tr = torch.load(filename, weights_only=False)
print(f"Training D:\t{endpoints_tr.shape}")  

filename = '/home/jacket/.cache/ikflow/datasets/panda/samples_tr.pt__tag0=non-self-colliding'
samples_tr = torch.load(filename, weights_only=False)
print(f"Training Q:\t{samples_tr.shape}") 

filename = '/home/jacket/.cache/ikflow/datasets/panda/endpoints_te.pt__tag0=non-self-colliding'
endpoints_te = torch.load(filename, weights_only=False)
print(f"Testing D:\t{endpoints_te.shape}")  

filename = '/home/jacket/.cache/ikflow/datasets/panda/samples_te.pt__tag0=non-self-colliding'
samples_te = torch.load(filename, weights_only=False)
print(f"Testing Q:\t{samples_te.shape}") 

# Ensure tensors are on CPU
endpoints_tr = endpoints_tr.cpu()
endpoints_te = endpoints_te.cpu()

# Extract first 3 dimensions
x1, y1, z1 = endpoints_tr[:, 0], endpoints_tr[:, 1], endpoints_tr[:, 2]
x2, y2, z2 = endpoints_te[:, 0], endpoints_te[:, 1], endpoints_te[:, 2]

# Create subplots
fig = plt.figure(figsize=(14, 6))

# Subplot 1: Training data
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(x1, y1, z1, color='blue', alpha=0.6, s=10)
ax1.set_title(f'Training Samples ({endpoints_tr.shape[0]})')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('Z-axis')

# Subplot 2: Testing data
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(x2, y2, z2, color='red', alpha=0.3, s=5)
ax2.set_title(f'Testing Samples ({endpoints_te.shape[0]})')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_zlabel('Z-axis')

#plt.tight_layout()
plt.show()