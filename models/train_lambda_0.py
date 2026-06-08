import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import MFPITDataset
from MFPIT_Model import MFPIT
from losses import MFPITLoss
from run_ablations import AblationDataset
from evaluate_all import two_stage_threshold_sweep

device = torch.device("cpu")

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    print("========================================================")
    # Scientific ablation: train MFPIT with lambda_phys = 0.0 (purely data-driven)
    print("--- LAUNCHING MFPIT ABLATION: LAMBDA_PHYS = 0.0 ---")
    print("========================================================")
    
    # 1. Load Datasets
    base_train = MFPITDataset(data_dir="../data/processed/tensors", split="train")
    base_val = MFPITDataset(data_dir="../data/processed/tensors", split="val")
    
    # Use "no_jrc" mode so JRC is zeroed out to prevent leakage
    train_dataset = AblationDataset(base_train, "no_jrc")
    val_dataset = AblationDataset(base_val, "no_jrc")
    
    g = torch.Generator()
    g.manual_seed(42)
    
    # We use batch_size = 8 and num_workers = 0 to match run_ablations.py settings exactly
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
    
    # Load physics stats
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
        
    # Scientific control: lambda_phys = 0.0, lambda_temp = 0.1 (same as control)
    criterion = MFPITLoss(
        lambda_phys=0.0, 
        lambda_temp=0.1, 
        pos_weight=pos_weight_val,
        balance_mu=balance_mu, 
        balance_sigma=balance_sigma
    )
    
    model = MFPIT().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    best_val_iou = -1.0
    out_dir = "engineering_validation_results"
    os.makedirs(out_dir, exist_ok=True)
    
    epochs = 8 # Train for 8 epochs (since best saved in epoch 4-10, 8 epochs is a solid sweet spot)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, phys_pred, phys_inputs, hidden_states = model(x)
            loss, loss_dict = criterion(logits, y, phys_pred, phys_inputs, hidden_states)
            
            loss.backward()
            optimizer.step()
            
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
                logits, phys_pred, phys_inputs, hidden_states = model(x)
                loss, _ = criterion(logits, y, phys_pred, phys_inputs, hidden_states)
                
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_probs.append(probs.view(-1).cpu().numpy())
                all_targets.append(y.view(-1).cpu().numpy())
                
        val_loss /= len(val_loader)
        all_probs_cat = np.concatenate(all_probs)
        all_targets_cat = np.concatenate(all_targets)
        
        best_t, val_iou = two_stage_threshold_sweep(all_probs_cat, all_targets_cat)
        print(f"Epoch {epoch+1} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Calibrated Val IoU: {val_iou:.4f} (Threshold: {best_t:.3f})")
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt_path = f"{out_dir}/ablation_lambda_0_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_iou': best_val_iou,
                'best_threshold': best_t,
            }, ckpt_path)
            print(f"--> Saved best checkpoint (Calibrated IoU: {best_val_iou:.4f}) to {ckpt_path}")
            
    print(f"\nLambda=0 training completed. Best Calibrated Val IoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    main()
