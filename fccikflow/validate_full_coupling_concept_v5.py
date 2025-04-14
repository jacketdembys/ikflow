import torch
from torch import nn, optim, distributions
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import os

# --- configuration --- #
BATCH_SIZE = 128
LOG_INTERVAL = 50
EPOCHS = 20
INPUT_DIM = 2
OUTPUT_DIM = 2
HIDDEN_DIM = 256
SAVE_PLT_INTERVAL = 5
N_COUPLE_LAYERS = 8
ARCH = 'symmetric'  # 'symmetric' or 'realnvp'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rcParams['figure.figsize'] = 8, 8
plt.ion()

# --- data loading --- #
train_data = datasets.make_moons(n_samples=50000, noise=.05)[0].astype(np.float32)
test_data = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# === Symmetric Coupling ===
class SymmetricFullCoupling(nn.Module):
    def __init__(self, dim, split_scheme, use_autograd_logdet=True, quat_start_idx = None):
        super().__init__()
        self.d1, self.d2 = split_scheme
        self.scale_limit = 1.0
        self.use_autograd_logdet = use_autograd_logdet
        self.quat_start_idx = quat_start_idx
        self.s1 = nn.Sequential(
            nn.Linear(self.d2, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, self.d1))
        self.t1 = nn.Sequential(
            nn.Linear(self.d2, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, self.d1))
        self.s2 = nn.Sequential(
            nn.Linear(self.d1, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, self.d2))
        self.t2 = nn.Sequential(
            nn.Linear(self.d1, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, self.d2))

    def _compute_autograd_logdet(self, x, rev=False):
        x = x.requires_grad_(True)
        log_dets = []
        for i in range(x.size(0)):
            inp = x[i].unsqueeze(0)
            out, _ = self.forward(inp, rev=rev, use_autograd_logdet=False)
            J = torch.autograd.functional.jacobian(lambda z: self.forward(z.unsqueeze(0), rev=rev, use_autograd_logdet=False)[0].squeeze(0), inp.squeeze(0))
            log_dets.append(torch.linalg.slogdet(J)[1])
        return torch.stack(log_dets, dim=0)

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
    

    def _normalize_quaternion_if_needed(self, x):
        if self.quat_start_idx is not None:
            pos = x[:, :self.quat_start_idx]
            quat = x[:, self.quat_start_idx:]
            quat = quat / (quat.norm(dim=1, keepdim=True) + 1e-8)
            return torch.cat([pos, quat], dim=1)
        return x



# === RealNVP Coupling Layer ===
class CouplingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask):
        super().__init__()
        self.s_fc1 = nn.Linear(input_dim, hid_dim)
        self.s_fc2 = nn.Linear(hid_dim, hid_dim)
        self.s_fc3 = nn.Linear(hid_dim, output_dim)
        self.t_fc1 = nn.Linear(input_dim, hid_dim)
        self.t_fc2 = nn.Linear(hid_dim, hid_dim)
        self.t_fc3 = nn.Linear(hid_dim, output_dim)
        self.mask = mask

    def forward(self, x):
        x_m = x * self.mask
        s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m))))))
        t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x_m)))))
        y = x_m + (1-self.mask)*(x*torch.exp(s_out)+t_out)
        log_det_jacobian = s_out.sum(dim=1)
        return y, log_det_jacobian

    def backward(self, y):
        y_m = y * self.mask
        s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(y_m))))))
        t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(y_m)))))
        x = y_m + (1-self.mask)*(y-t_out)*torch.exp(-s_out)
        return x

# === RealNVP Model ===
class RealNVP(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask, n_layers = 6):
        super().__init__()
        assert n_layers >= 2, 'num of coupling layers should be greater or equal to 2'
        self.modules = []
        self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, mask))
        for _ in range(n_layers-2):
            mask = 1 - mask
            self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, mask))
        self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, 1 - mask))
        self.module_list = nn.ModuleList(self.modules)

    def forward(self, x):
        ldj_sum = 0
        for module in self.module_list:
            x, ldj = module(x)
            ldj_sum += ldj
        return x, ldj_sum

    def backward(self, z):
        for module in reversed(self.module_list):
            z = module.backward(z)
        return z

# === Architecture selector ===
class SymmetricStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([SymmetricFullCoupling(INPUT_DIM, [1, 1]) for _ in range(N_COUPLE_LAYERS)])
    def forward(self, x):
        log_det = 0
        for layer in self.layers:
            x, ld = layer(x, rev=False)
            log_det += ld
        return x, log_det
    def backward(self, z):
        for layer in reversed(self.layers):
            z, _ = layer(z, rev=True)
        return z

def get_model(arch='realnvp'):
    if arch == 'realnvp':
        mask = torch.from_numpy(np.array([0, 1]).astype(np.float32)).to(device)
        return RealNVP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, mask, N_COUPLE_LAYERS)
    elif arch == 'symmetric':
        return SymmetricStack()
    else:
        raise ValueError("Unknown architecture")

model = get_model(ARCH).to(device)
prior_z = distributions.MultivariateNormal(torch.zeros(2, device=device), torch.eye(2, device=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Train and Test ---
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        z, log_det_j_sum = model(data)
        loss = -(prior_z.log_prob(z)+log_det_j_sum).mean()
        loss.backward()
        cur_loss = loss.item()
        train_loss += cur_loss
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {cur_loss/len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(epoch):
    model.eval()
    test_loss = 0
    x_all = np.array([[]]).reshape(0,2)
    z_all = np.array([[]]).reshape(0,2)
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            z, log_det_j_sum = model(data)
            cur_loss = -(prior_z.log_prob(z)+log_det_j_sum).mean().item()
            test_loss += cur_loss
            x_all = np.concatenate((x_all,data.cpu().numpy()))
            z_all = np.concatenate((z_all,z.cpu().numpy()))
        subfig_plot(1, x_all, -2, 3, -1, 1.5,'Input: x ~ p(x)', 'b')
        subfig_plot(2, z_all, -3, 3, -3,3,'Output: z = f(x)', 'b')
        test_loss /= len(test_loader.dataset)
        print(f'====> Test set loss: {test_loss:.4f}')

from sklearn.metrics import mean_squared_error

def sample(epoch):
    model.eval()
    with torch.no_grad():
        z = prior_z.sample((1000,)).to(device)
        x = model.backward(z)
        z = z.cpu().numpy()
        x = x.cpu().numpy()
        
        subfig_plot(3, z, -3, 3, -3, 3, 'Input: z ~ p(z)', 'r')
        subfig_plot(4, x, -2, 3, -1, 1.5,'Output: x = g(z)', 'r')
        if epoch % SAVE_PLT_INTERVAL == 0:
            os.makedirs('results', exist_ok=True)
            plt.savefig(f'results/result_{ARCH}_{epoch}.pdf')

def subfig_plot(location, data, x_start, x_end, y_start, y_end, title, color):
    if location == 1:
        plt.clf()
    plt.subplot(2,2,location)
    plt.scatter(data[:, 0], data[:, 1], c=color, s=1)
    plt.xlim(x_start,x_end)
    plt.ylim(y_start,y_end)
    plt.title(title)
    plt.pause(1e-2)

# === Main ===
if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
        sample(epoch)
