import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from jrl.robots import get_robot
from ikflow.evaluation_utils import (
    _get_target_pose_batch,
    solution_pose_errors,
    calculate_joint_limits_exceeded,
    calculate_self_collisions,
)

# Dataset loader
class DiffIKDataset(Dataset):
    def __init__(self, filename_D, filename_Q):
        self.q = torch.from_numpy(np.load(filename_Q)).float()
        self.pose = torch.from_numpy(np.load(filename_D)).float()
        assert self.q.shape[0] == self.pose.shape[0], "Mismatch in sample count"

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return {
            "q": self.q[idx],
            "pose": self.pose[idx],
        }

# Diffusion Architecture
class DiffIKDenoiser(nn.Module):
    def __init__(self, dof=7, pose_dim=7, hidden_dim=512, time_embed_dim=64):
        super().__init__()
        self.dof = dof
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        input_dim = dof + pose_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, dof)
        )

    def forward(self, q_t, pose, t):
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        x = torch.cat([q_t, pose, t_embed], dim=-1)
        return self.net(x)

def sample(model, pose, num_timesteps=1000, ddim_steps=50, n_samples=1):
    model.eval()
    B = pose.shape[0]
    q = torch.randn(B * n_samples, model.dof, device=pose.device)
    pose = pose.repeat_interleave(n_samples, dim=0)
    for t in reversed(range(1, ddim_steps + 1)):
        t_tensor = torch.ones(q.size(0), device=q.device).long() * t
        beta = 1e-4 + (t_tensor / num_timesteps).unsqueeze(-1)
        noise_pred = model(q, pose, t_tensor)
        q = q - beta * noise_pred
    return q

def validate(model, loader, robot, device):
    model.eval()
    total_pos_err, total_ori_err = 0.0, 0.0
    for batch in loader:
        q_gt = batch["q"].to(device)
        pose_gt = batch["pose"].to(device)
        q_pred = sample(model, pose_gt)
        target_poses = _get_target_pose_batch(pose_gt, q_pred.shape[0])
        l2_errors, ang_errors = solution_pose_errors(robot, q_pred, target_poses)
        total_pos_err += l2_errors.mean().item() * 1000
        total_ori_err += ang_errors.mean().item() * (180.0 / 3.14159)
    return total_pos_err / len(loader), total_ori_err / len(loader)

def train_loop(model, train_loader, val_loader, robot, max_epochs=100, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    print("Using device:", device)

    for epoch in range(max_epochs):
        print(f"\n[Epoch {epoch}]")
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            try:
                q = batch["q"].to(device)
                pose = batch["pose"].to(device)
                noise = torch.randn_like(q)
                t = torch.randint(0, model.num_timesteps, (q.size(0),), device=q.device)
                beta = 1e-4 + (t / model.num_timesteps).unsqueeze(-1)
                q_t = q + beta * noise
                noise_pred = model(q_t, pose, t)
                loss = loss_fn(noise_pred, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"    ✅ Loss: {loss.item():.6f}")
                break  # Debug first batch only
            except Exception as e:
                print("🔥 ERROR DURING BATCH PROCESSING:", str(e))
                import traceback
                traceback.print_exc()
                return

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Configurable Parameters ---
    batch_size = 128
    dof = 7
    pose_dim = 7

    robot = get_robot("panda")

    # Load datasets
    filename_Dtr = '/home/jacket/Documents/ikflow/datasets/panda/endpoints_tr.npy'
    filename_Qtr = '/home/jacket/Documents/ikflow/datasets/panda/samples_tr.npy'
    train_dataset = DiffIKDataset(filename_Dtr, filename_Qtr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    filename_Dte = '/home/jacket/Documents/ikflow/datasets/panda/endpoints_te.npy'
    filename_Qte = '/home/jacket/Documents/ikflow/datasets/panda/samples_te.npy'
    val_dataset = DiffIKDataset(filename_Dte, filename_Qte)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    model = DiffIKDenoiser(dof=dof, pose_dim=pose_dim)
    model.dof = dof
    model.num_timesteps = 1000

    train_loop(model, train_loader, val_loader, robot)
