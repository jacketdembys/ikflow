import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from jrl.robots import Robot

import sys
import os
import torch.multiprocessing as mp



### Class Definitions

# Dataset loader
class DiffIKDataset(Dataset):
    def __init__(self, filename_D, filename_Q):
        self.q = torch.load(filename_Q, weights_only=False).cpu()          # shape: [N, dof]
        self.pose = torch.load(filename_D, weights_only=False).cpu()     # shape: [N, 6 or 7]
        assert self.q.shape[0] == self.pose.shape[0]

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):

        q_item = self.q[idx]
        pose_item = self.pose[idx]
        
        print(f"[Dataset] q.device: {q_item.device}, pose.device: {pose_item.device}")

        return {
            "q": self.q[idx],        # joint configuration
            "pose": self.pose[idx],  # end-effector pose
        }


# Diffusion Architecture MLP-based
class DiffIKDenoiser(nn.Module):
    def __init__(self, dof=7, pose_dim=7, hidden_dim=512, time_embed_dim=64):
        super().__init__()
        self.dof = dof
        self.pose_dim = pose_dim
        self.time_embed_dim = time_embed_dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        input_dim = dof + pose_dim + time_embed_dim

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, dof)
        )

    def forward(self, q_t, pose, t):
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        x = torch.cat([q_t, pose, t_embed], dim=-1)
        return self.net(x)


# Pytorch Lightning Module for fast prototyping
class DiffIKLightningModule(pl.LightningModule):
    def __init__(
        self,
        dof=7,
        pose_dim=7,
        hidden_dim=256,
        time_embed_dim=64,
        num_timesteps=1000,
        learning_rate=1e-4,
        robot=None
    ):
        super().__init__()
        #self.save_hyperparameters()
        self.denoiser = DiffIKDenoiser(
            dof=dof,
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
        )
        self.num_timesteps = num_timesteps
        self.loss_fn = nn.MSELoss()
        self.robot = robot
        self.dof = dof
        self.pose_dim = pose_dim
        self.learning_rate = learning_rate

    def forward(self, q_t, pose, t):
        return self.denoiser(q_t, pose, t)

    def training_step(self, batch, batch_idx):
        print("\nHere")
        q = batch["q"]                  # [B, dof]
        pose = batch["pose"]            # [B, 7]
        noise = torch.randn_like(q)     # ε ~ N(0, I)
        #print(q.device, pose.device, noise.device)
        print("training_step device check:", q.device, pose.device)
        #sys.exit()
        #t = torch.randint(0, self.num_timesteps, (q.size(0),), device=q.device)
        """
        gen = torch.Generator(device=q.device)  # <- force generator to use the correct device
        t = torch.randint(0, self.num_timesteps, (q.size(0),), device=q.device, generator=gen)
        """
        t = (torch.rand(q.size(0), device=q.device) * self.num_timesteps).long()



        # Linearly scale noise level (β schedule)
        beta = 1e-4 + (t / self.num_timesteps).unsqueeze(-1)  # shape: [B, 1]
        q_t = q + beta * noise                                # q_t = q + scaled noise

        noise_pred = self(q_t, pose, t)
        loss = self.loss_fn(noise_pred, noise)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


    @torch.no_grad()
    def sample(self, pose, n_samples=1, ddim_steps=50):
        """
        pose: [1, 7] or [B, 7] – end-effector pose
        returns: [B * n_samples, dof] – predicted joint configurations
        """
        self.eval()
        B = pose.shape[0]
        q = torch.randn(B * n_samples, self.dof).to(pose.device)
        pose = pose.repeat_interleave(n_samples, dim=0)

        print("sample_step device check:", q.device, pose.device)
        
        for t in reversed(range(1, ddim_steps + 1)):
            #t_tensor = torch.full((q.size(0),), t, device=q.device)
            t_tensor = torch.ones(q.size(0), device=q.device).long() * t
            #print("t_tensor device:", t_tensor.device)
            noise_pred = self(q, pose, t_tensor)
            beta = 1e-4 + (t_tensor / self.num_timesteps).unsqueeze(-1)
            q = q - beta * noise_pred  # DDIM-like update (simplified)

        print("sample_step device check done\n")
        return q  # final predicted joint configuration


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        q_gt = batch["q"]           # Ground-truth joints
        pose_gt = batch["pose"]     # Ground-truth pose
        B = q_gt.shape[0]

        # Sample predicted joint config from the model
        q_pred = self.sample(pose_gt, n_samples=1)  # shape: [B, dof]

        print("validation_step device check:", q_gt.device, pose_gt.device, q_pred.device)

        from ikflow.evaluation_utils import _get_target_pose_batch, solution_pose_errors, calculate_joint_limits_exceeded, calculate_self_collisions 

        target_poses = _get_target_pose_batch(pose_gt, q_pred.shape[0])
        l2_errors, ang_errors = solution_pose_errors(self.robot, q_pred, target_poses)
        joint_limits_exceeded = calculate_joint_limits_exceeded(q_pred, self.robot.actuated_joints_limits)
        self_collisions_respected = calculate_self_collisions(self.robot, q_pred)

        self.log("val_pos_err_mm", (l2_errors.mean() * 1000.0).detach().cpu().item(), prog_bar=True)
        self.log("val_ori_err_deg", (ang_errors.mean() * (180.0 / 3.14159)).detach().cpu().item(), prog_bar=True)

        val_pose_err = l2_errors.mean()*1000 + ang_errors.mean()* (180.0 / 3.14159)
        self.log("val_pose_err", (l2_errors.mean()*1000 + ang_errors.mean()* (180.0 / 3.14159)).detach().cpu().item())

        print("validation_step device check done\n")
        val_pose_err = val_pose_err.detach().cpu().item()
        return val_pose_err


### Herlper functions
def safe_worker_init_fn(worker_id):
    # This avoids accidental CUDA context in workers
    worker_info = torch.utils.data.get_worker_info()
    #torch.manual_seed(worker_info.seed)



### Main function
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # --- Configurable Parameters ---
    batch_size = 128
    max_epochs = 100
    disable_wandb = False  # Set True if using Weights & Biases
    dof = 7
    pose_dim = 7

    from jrl.robots import get_robot, Robot
    robot_name = "panda"
    robot = get_robot(robot_name)


    # --- Setup Wandb logging ---
    wandb_logger = None
    if not disable_wandb:

        # Make sure any previous run is closed
        if wandb.run is not None:
            wandb.finish()

        # Initialize logger (let WandbLogger handle wandb.init)
        wandb_logger = WandbLogger(
            entity="jacketdembys",
            project="diffik",
            log_model="all",
            group="MLP_"+robot_name+"_2500000",
            name="MLP_"+robot_name+"_2500000_BS_"+str(batch_size)+"_Optimizer_AdamW",
            save_dir="wandb_logs"
        )



    # --- Load Full Dataset ---
    num_workers = os.cpu_count() if torch.cuda.is_available() else 0
    num_workers = 0
    print(f"Using: {num_workers} workers")

    filename_Dtr = '/home/jacket/.cache/ikflow/datasets/panda/endpoints_tr.pt__tag0=non-self-colliding'
    filename_Qtr = '/home/jacket/.cache/ikflow/datasets/panda/samples_tr.pt__tag0=non-self-colliding'
    training_dataset = DiffIKDataset(filename_Dtr, filename_Qtr)
    training_loader = DataLoader(training_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    #worker_init_fn=safe_worker_init_f
                                    )

    filename_Dte = '/home/jacket/.cache/ikflow/datasets/panda/endpoints_te.pt__tag0=non-self-colliding'
    filename_Qte = '/home/jacket/.cache/ikflow/datasets/panda/samples_te.pt__tag0=non-self-colliding'
    validation_dataset = DiffIKDataset(filename_Dte, filename_Qte)
    validation_loader = DataLoader(validation_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    #worker_init_fn=safe_worker_init_f
                                    )

    # --- Model & Checkpointing ---
    model = DiffIKLightningModule(dof=dof, 
                                    pose_dim=pose_dim, 
                                    robot=robot)
    print(model.denoiser)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_pose_err",
        mode="min",
        save_top_k=3,
        filename="best-diffik-{epoch:02d}-{val_pose_err:.2f}",
        auto_insert_metric_name=False,
        save_weights_only=False,
    )

    # --- Trainer ---
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        default_root_dir=f"results/diffik/{robot_name}"
    )

    # --- Training ---
    
    trainer.fit(model, 
                train_dataloaders=training_loader, 
                val_dataloaders=validation_loader)
    

    