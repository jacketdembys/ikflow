import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import numpy as np

# === Coupling Layer Registry ===
class SymmetricFullCoupling(nn.Module):
    def __init__(self, dim, split_scheme, subnet_constructor, quat_start_idx=None):
        super().__init__()
        assert sum(split_scheme) == dim, "Split scheme must match total dimension"
        self.scale_limit = 1.0  # Clamp s(x) outputs to ±1.0 for stability
        self.dim = dim
        self.d1, self.d2 = split_scheme
        self.quat_start_idx = quat_start_idx

        self.s1 = subnet_constructor(self.d2, self.d1)
        self.t1 = subnet_constructor(self.d2, self.d1)
        self.s2 = subnet_constructor(self.d1, self.d2)
        self.t2 = subnet_constructor(self.d1, self.d2)

    def forward(self, x, rev=False, use_autograd_logdet=True, use_trace_estimate=False):
        x1, x2 = x[:, :self.d1], x[:, self.d1:]

        if not rev:
            # Clamp the scale output to avoid extreme compression or expansion
            s2 = torch.tanh(self.s1(x2)) * self.scale_limit  # updated clamp scale_limit = ±1.0  # scale limit = 2.0
            t2 = self.t1(x2)
            y1 = x1 * torch.exp(s2) + t2

            s1 = torch.tanh(self.s2(y1)) * self.scale_limit  # updated clamp scale_limit = ±1.0  # scale limit = 2.0
            t1 = self.t2(y1)
            y2 = x2 * torch.exp(s1) + t1

            y = torch.cat([y1, y2], dim=1)
            y = self._normalize_quaternion_if_needed(y)
            if use_trace_estimate:
                log_det = self._compute_trace_estimate_logdet(x, rev=rev)
            elif use_autograd_logdet:
                log_det = self._compute_autograd_logdet(x, rev=False)
            else:
                log_det = s2.sum(dim=1) + s1.sum(dim=1)
        else:
            s1 = torch.tanh(self.s2(x[:, :self.d1])) * self.scale_limit  # updated clamp scale_limit = ±1.0  # scale limit = 2.0
            t1 = self.t2(x[:, :self.d1])
            x2 = (x[:, self.d1:] - t1) * torch.exp(-s1)

            s2 = torch.tanh(self.s1(x2)) * self.scale_limit  # updated clamp scale_limit = ±1.0  # scale limit = 2.0
            t2 = self.t1(x2)
            x1 = (x[:, :self.d1] - t2) * torch.exp(-s2)

            y = torch.cat([x1, x2], dim=1)
            y = self._normalize_quaternion_if_needed(y)
            if use_autograd_logdet:
                log_det = self._compute_autograd_logdet(x, rev=True)
            else:
                log_det = -(s1.sum(dim=1) + s2.sum(dim=1))

        return y, log_det

    def _compute_autograd_logdet(self, x, rev=False):
        x = x.requires_grad_(True)
        log_dets = []
        for i in range(x.size(0)):
            inp = x[i].unsqueeze(0)
            out, _ = self.forward(inp, rev=rev, use_autograd_logdet=False)
            J = torch.autograd.functional.jacobian(lambda z: self.forward(z.unsqueeze(0), rev=rev, use_autograd_logdet=False)[0].squeeze(0), inp.squeeze(0))
            log_dets.append(torch.linalg.slogdet(J)[1])
        return torch.stack(log_dets, dim=0)

    def _compute_trace_estimate_logdet(self, x, rev=False, num_samples=1):
        x = x.detach().requires_grad_(True)
        log_dets = []
        for i in range(x.size(0)):
            inp = x[i].unsqueeze(0)
            def f(z):
                return self.forward(z.unsqueeze(0), rev=rev, use_autograd_logdet=False)[0].squeeze(0)
            trace_estimates = []
            for _ in range(num_samples):
                v = torch.randn_like(inp.squeeze(0))
                jvp = torch.autograd.functional.jvp(f, inp.squeeze(0), v=v, create_graph=False)[1]
                trace_estimates.append(torch.dot(v, jvp))
            trace_avg = torch.stack(trace_estimates).mean()
            log_dets.append(trace_avg)
        return torch.stack(log_dets, dim=0)

    def _normalize_quaternion_if_needed(self, x):
        if self.quat_start_idx is not None:
            pos = x[:, :self.quat_start_idx]
            quat = x[:, self.quat_start_idx:]
            quat = quat / (quat.norm(dim=1, keepdim=True) + 1e-8)
            return torch.cat([pos, quat], dim=1)
        return x

class RealNVPCoupling(nn.Module):
    def __init__(self, dim, split_scheme, subnet_constructor):
        super().__init__()
        assert sum(split_scheme) == dim, "Split scheme must match total dimension"
        self.dim = dim
        self.d1, self.d2 = split_scheme

        self.s = subnet_constructor(self.d1, self.d2)
        self.t = subnet_constructor(self.d1, self.d2)

    def forward(self, x, rev=False, use_autograd_logdet=False, use_trace_estimate=False):
        x1, x2 = x[:, :self.d1], x[:, self.d1:]
        if not rev:
            s_val = self.s(x1)
            t_val = self.t(x1)
            y2 = x2 * torch.exp(s_val) + t_val
            y = torch.cat([x1, y2], dim=1)
            log_det = s_val.sum(dim=1)
        else:
            s_val = self.s(x1)
            t_val = self.t(x1)
            x2 = (x2 - t_val) * torch.exp(-s_val)
            y = torch.cat([x1, x2], dim=1)
            log_det = -s_val.sum(dim=1)
        return y, log_det

def make_mlp(in_dim, out_dim, hidden=64):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim)
    )

# === Architecture Selector ===
coupling_layers = {
    'symmetric_full': SymmetricFullCoupling,
    'realnvp': RealNVPCoupling,
}

def get_model(architecture='symmetric_full', dim=2, split_scheme=[1, 1]):
    if architecture not in coupling_layers:
        raise ValueError(f"Unknown architecture: {architecture}")
    model_class = coupling_layers[architecture]
    return model_class(dim=dim, split_scheme=split_scheme, subnet_constructor=make_mlp)

def train_on_two_moons(architecture='symmetric_full', save_path='trained_model.pth', logdet_type='autograd'):
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)

    model = get_model(architecture=architecture, dim=2, split_scheme=[1, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    best_model_state = None

    print_interval = 10
    start_time = time.time()

    logpz_history = []
    logdet_history = []
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for batch, in train_loader:
            z, logdet = model(batch, use_autograd_logdet=(logdet_type=='autograd'), use_trace_estimate=(logdet_type=='trace'))
            log_pz = -0.5 * torch.sum(z ** 2, dim=1) - torch.log(torch.tensor(2 * torch.pi)).item()
            logpz_history.append(log_pz.mean().item())
            logdet_history.append(logdet.mean().item())
            # Apply regularization penalty on logdet to prevent volume collapse
            logdet_penalty = 1e-2 * torch.mean(logdet ** 2)  # You can tune this coefficient
            loss = -torch.mean(log_pz + logdet) + logdet_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        model.eval()
        with torch.no_grad():
            z_val, logdet_val = model(X_val, use_autograd_logdet=(logdet_type=='autograd'), use_trace_estimate=(logdet_type=='trace'))
            log_pz_val = -0.5 * torch.sum(z_val ** 2, dim=1) - torch.log(torch.tensor(2 * torch.pi)).item()
            val_loss = -torch.mean(log_pz_val + logdet_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if epoch % print_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {total_loss / len(train_loader.dataset):.4f} | Val Loss: {val_loss:.4f} | Avg log_pz: {log_pz.mean().item():.4f} | Avg logdet: {logdet.mean().item():.4f} | Logdet penalty: {logdet_penalty.item():.4e}")

    train_time = time.time() - start_time
    print(f"Training time for {architecture}: {train_time:.2f} seconds")

    if best_model_state is not None:
        torch.save(best_model_state, save_path)
        print(f"Best model saved to {save_path}")

    # Plot log_pz and logdet over time
    plt.figure(figsize=(10, 5))
    plt.plot(logpz_history, label='avg log_pz')
    plt.plot(logdet_history, label='avg logdet')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'log_pz and logdet over Epochs ({architecture}, {logdet_type})')
    plt.legend()
    plt.tight_layout()
    log_plot_path = f"log_terms_{architecture}_{logdet_type}.pdf"
    plt.savefig(log_plot_path)
    print(f"Log term plot saved to {log_plot_path}")
    plt.close()

def visualize_two_moons_model(architecture='symmetric_full', model_path='trained_model.pth', logdet_type='autograd'):
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)

    model = get_model(architecture=architecture, dim=2, split_scheme=[1, 1])
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        start_eval = time.time()
        with torch.no_grad():
            z, _ = model(X, use_autograd_logdet=(logdet_type=='autograd'), use_trace_estimate=(logdet_type=='trace'))
            x_recon, _ = model(z, rev=True, use_autograd_logdet=(logdet_type=='autograd'), use_trace_estimate=(logdet_type=='trace'))
        eval_time = time.time() - start_eval
        print(f"Evaluation time for {architecture}: {eval_time:.2f} seconds")

        X_np = X.detach().numpy()
        x_recon_np = x_recon.detach().numpy()
        recon_error = mean_squared_error(X_np, x_recon_np)
        cm = confusion_matrix(y.numpy(), y.numpy())
        print(f"Class correspondence (confusion-matrix-style):\n{cm}")

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        labels = ['Class 0', 'Class 1']
        colors = ['blue', 'red']
        for i, label in enumerate(labels):
            axs[0].scatter(X_np[y == i][:, 0], X_np[y == i][:, 1], c=colors[i], s=10, label=label)
        axs[0].set_title("Original Data")
        for i, label in enumerate(labels):
            z_np = z.detach().numpy()
        axs[1].scatter(z_np[y == i][:, 0], z_np[y == i][:, 1], c=colors[i], s=10, label=label)
        axs[1].set_title("Mapped to Latent Space")
        for i, label in enumerate(labels):
            axs[2].scatter(x_recon_np[y == i][:, 0], x_recon_np[y == i][:, 1], c=colors[i], s=10, label=label)
        axs[2].set_title(f"Reconstructed\nMSE: {recon_error:.4e}")
        for ax in axs:
            ax.legend()
        plt.tight_layout()

        fig_path = f"fig_{architecture}_{logdet_type}.pdf"
        plt.savefig(fig_path)
        print(f"Visualization saved to {fig_path}")
        plt.show()
    else:
        print(f"Model path {model_path} does not exist.")



if __name__ == '__main__':
    architectures_to_train = ['symmetric_full'] # ['symmetric_full', 'realnvp']
    logdet_calculation_method = ['autograd'] #['autograd', 'trace', 'approx']

    for arch in architectures_to_train:
        for logdet_type in logdet_calculation_method:
            model_file = f"{arch}_model_{logdet_type}.pth"
            print(f"model file naming: {model_file}")
            train_on_two_moons(architecture=arch, save_path=model_file, logdet_type=logdet_type)
            visualize_two_moons_model(architecture=arch, model_path=model_file, logdet_type=logdet_type)
