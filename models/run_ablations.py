import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import MFPITDataset
from metrics import compute_flat_metrics
from MFPIT_Model import MFPIT
from losses import MFPITLoss
from evaluate_all import two_stage_threshold_sweep

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AblationDataset(torch.utils.data.Dataset):
    """
    Wrapper around MFPITDataset that zeros out specific feature channels
    to implement the scientific ablation experiments.
    """
    def __init__(self, base_dataset, mode):
        self.base_dataset = base_dataset
        self.mode = mode
        print(f"Initializing Ablation Dataset in mode: {mode}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx] # x: [12, 13, 64, 64], y: [1, 64, 64]
        x_ablated = x.clone()

        if self.mode == "static_full":
            # Keep ONLY static channels: DEM elevation (8), DEM slope (9), and JRC occurrence (12)
            # Zero out all other channels (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)
            keep_indices = [8, 9, 12]
            for c in range(13):
                if c not in keep_indices:
                    x_ablated[:, c, :, :] = 0.0

        elif self.mode == "terrain_only":
            # Keep ONLY geographic terrain features: DEM elevation (8) and DEM slope (9)
            # Zero out everything else, including JRC water occurrence (12)
            keep_indices = [8, 9]
            for c in range(13):
                if c not in keep_indices:
                    x_ablated[:, c, :, :] = 0.0

        elif self.mode == "no_jrc":
            # Keep everything EXCEPT JRC occurrence (12)
            x_ablated[:, 12, :, :] = 0.0

        elif self.mode == "dynamic_only":
            # Keep ONLY dynamic channels (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)
            # Zero out all static channels: DEM (8), Slope (9), JRC occurrence (12)
            static_indices = [8, 9, 12]
            for c in static_indices:
                x_ablated[:, c, :, :] = 0.0

        elif self.mode == "hydro_only":
            # Keep ONLY precipitation (2), ET (3), Temp (4, 5), Soil (6), Runoff (7)
            keep_indices = [2, 3, 4, 5, 6, 7]
            for c in range(13):
                if c not in keep_indices:
                    x_ablated[:, c, :, :] = 0.0

        elif self.mode == "eo_only":
            # Keep ONLY Optical (0, 1) and SAR (10, 11)
            keep_indices = [0, 1, 10, 11]
            for c in range(13):
                if c not in keep_indices:
                    x_ablated[:, c, :, :] = 0.0

        return x_ablated, y

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_ablation(mode, epochs=50):
    print(f"\n========================================================")
    print(f"--- LAUNCHING MFPIT ABLATION EXPERIMENT: {mode.upper()} ---")
    print(f"========================================================")

    # 1. Load Datasets
    base_train = MFPITDataset(data_dir="../data/processed/tensors", split="train")
    base_val = MFPITDataset(data_dir="../data/processed/tensors", split="val")

    train_dataset = AblationDataset(base_train, mode)
    val_dataset = AblationDataset(base_val, mode)

    # 2. PyTorch DataLoaders with reproducibility seeds
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g
    )

    # 3. Load frozen physics stats & setup loss
    phys_stats_file = "physics_stats.json"
    if os.path.exists(phys_stats_file):
        with open(phys_stats_file, "r") as f:
            stats = json.load(f)
            balance_mu = stats['balance_mu']
            balance_sigma = stats['balance_sigma']
            pos_weight_val = stats.get('pos_weight', 5.0)
    else:
        balance_mu = 0.0
        balance_sigma = 1.0
        pos_weight_val = 5.0

    # Reviewer-grade check: disable physics loss for static/eo modes to avoid loss mismatch
    # Disable temporal consistency loss for purely static/terrain experiments to prevent artificial continuity penalties
    lambda_phys_val = 0.0 if mode in ["static_full", "terrain_only", "eo_only"] else 0.1
    lambda_temp_val = 0.0 if mode in ["static_full", "terrain_only"] else 0.1
    print(f"Loss Configuration: lambda_phys={lambda_phys_val:.2f} | lambda_temp={lambda_temp_val:.2f} (Ablation: {mode})")

    criterion = MFPITLoss(
        lambda_phys=lambda_phys_val, 
        lambda_temp=lambda_temp_val, 
        pos_weight=pos_weight_val,
        balance_mu=balance_mu, 
        balance_sigma=balance_sigma
    )

    model = MFPIT().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # 4. Safer CPU/CUDA AMP setup
    use_amp = torch.cuda.is_available()
    device_type = 'cuda' if use_amp else 'cpu'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_iou = -1.0
    out_dir = "engineering_validation_results"
    os.makedirs(out_dir, exist_ok=True)
    patience = 5
    epochs_no_improve = 0

    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                logits, phys_pred, phys_inputs, hidden_states = model(x)
                loss, loss_dict = criterion(logits, y, phys_pred, phys_inputs, hidden_states)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss_dict['total'], 'bce': loss_dict['bce'], 'dice': loss_dict['dice']})
            
        train_loss /= len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    logits, phys_pred, phys_inputs, hidden_states = model(x)
                    loss, _ = criterion(logits, y, phys_pred, phys_inputs, hidden_states)
                
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_probs.append(probs.view(-1).cpu().numpy())
                all_targets.append(y.view(-1).cpu().numpy())
                
        val_loss /= len(val_loader)
        all_probs_cat = np.concatenate(all_probs)
        all_targets_cat = np.concatenate(all_targets)
        
        # Fair Early Stopping: sweep decision thresholds over validation predictions
        best_t, val_iou = two_stage_threshold_sweep(all_probs_cat, all_targets_cat)
        print(f"Epoch {epoch+1} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Calibrated Val IoU: {val_iou:.4f} (Threshold: {best_t:.3f})")
        
        # Checkpoint Best Model based on Calibrated Val IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            ckpt_path = f"{out_dir}/ablation_{mode}_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_iou': best_val_iou,
                'best_threshold': best_t,
            }, ckpt_path)
            print(f"--> Saved best checkpoint (Calibrated IoU: {best_val_iou:.4f}) to {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in Val IoU for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs! Best Val IoU: {best_val_iou:.4f}")
                break

    print(f"\nAblation {mode.upper()} completed. Best Calibrated Val IoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scientific MFPIT feature ablations.")
    parser.add_argument(
        "--ablation", 
        type=str, 
        required=True, 
        choices=["static_full", "terrain_only", "no_jrc", "dynamic_only", "hydro_only", "eo_only"],
        help="Ablation mode to run."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    args = parser.parse_args()
    
    train_ablation(args.ablation, epochs=args.epochs)
