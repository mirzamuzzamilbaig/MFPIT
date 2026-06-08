import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import json
from dataset import MFPITDataset
from MFPIT_Model import MFPIT, CHANNEL_MAP
from losses import MFPITLoss
from metrics import compute_flat_metrics

def set_seed(strict_repro=False):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if strict_repro:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("Reproducibility Mode: STRICT (Deterministic, slower)")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        print("Reproducibility Mode: PERFORMANCE (Benchmark enabled)")

def compute_label_prevalence(dataset):
    print("Scanning dataset for label prevalence to configure pos_weight...")
    # Scan a larger subset of 1000 patches for stable estimation
    sample_size = min(1000, len(dataset))
    water_pixels = 0
    total_pixels = 0
    
    # Reproducible random seed for sampling to eliminate temporal/spatial bias
    np.random.seed(42)
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for i in indices:
        _, y = dataset[int(i)]
        water_pixels += y.sum().item()
        total_pixels += y.numel()
        
    water_pct = water_pixels / (total_pixels + 1e-8)
    non_water_pct = 1.0 - water_pct
    
    print(f"  Water: {water_pct*100:.2f}% | Non-Water: {non_water_pct*100:.2f}%")
    
    if water_pct < 1e-5:
        # Failsafe if completely dry
        return 1.0
        
    # pos_weight = negative_samples / positive_samples
    pos_weight = non_water_pct / water_pct
    return min(pos_weight, 50.0) # Cap at 50 to prevent explosion

def dry_run_hardware_check(model, device, use_amp):
    print("\n--- Dry Run Hardware Verification ---")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f} M")
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        
        dummy_x = torch.randn(2, 12, 13, 64, 64, device=device)
        dummy_y = torch.randint(0, 2, (2, 1, 64, 64), device=device).float()
        
        # JRC disabled
        dummy_x[:, :, 12, :, :] = 0.0
        
        autocast_context = torch.amp.autocast('cuda') if use_amp else torch.amp.autocast('cpu', enabled=False)
        with autocast_context:
            logits, phys_pred, phys_inputs, hidden_states = model(dummy_x)
            
        loss = logits.sum() # Dummy loss for backward memory test
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else torch.cuda.amp.GradScaler(enabled=False)
        scaler.scale(loss).backward()
        
        max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"Estimated Max VRAM (Batch=2, Patch=64x64): {max_mem_mb:.2f} MB")
        
        # Clear dummy data from VRAM
        model.zero_grad()
        del dummy_x, dummy_y, logits, loss
        torch.cuda.empty_cache()
    else:
        print("CUDA not available. Skipping VRAM measurement.")
    print("--------------------------------------\n")

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train():
    strict_repro = True # Enable for exact publication reproducibility
    set_seed(strict_repro)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else torch.cuda.amp.GradScaler(enabled=False)
    
    batch_size = 4
    epochs = 50
    lr = 1e-4
    
    print("Initializing datasets... [ENGINEERING VALIDATION MODE]")
    train_dataset = MFPITDataset(data_dir="../data/processed/tensors", split="train")
    val_dataset = MFPITDataset(data_dir="../data/processed/tensors", split="val")
    
    if len(train_dataset) == 0:
        print("Waiting for data extraction and preprocessing to complete.")
        # But we can still run dry run using dummy model if needed
    else:
        pos_weight_val = compute_label_prevalence(train_dataset)
        print(f"Computed pos_weight: {pos_weight_val:.2f}")
    
    # Load frozen physics stats
    phys_stats_file = "physics_stats.json"
    if os.path.exists(phys_stats_file):
        with open(phys_stats_file, "r") as f:
            stats = json.load(f)
            balance_mu = stats['balance_mu']
            balance_sigma = stats['balance_sigma']
            print(f"Loaded frozen physics stats: mu={balance_mu:.4f}, sigma={balance_sigma:.4f}")
    else:
        print("Warning: physics_stats.json not found. Using defaults. Run compute_physics_stats.py first.")
        balance_mu = 0.0
        balance_sigma = 1.0
        pos_weight_val = 5.0 # Fallback
        
    model = MFPIT().to(device)
    
    dry_run_hardware_check(model, device, use_amp)
    
    if len(train_dataset) == 0:
        return
        
    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    
    criterion = MFPITLoss(lambda_phys=0.1, lambda_temp=0.1, pos_weight=pos_weight_val, 
                          balance_mu=balance_mu, balance_sigma=balance_sigma)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    out_dir = "engineering_validation_results"
    os.makedirs(out_dir, exist_ok=True)
    best_val_iou = -1.0 # Early stopping on IoU
    
    # Save config
    config = {
        'strict_repro': strict_repro,
        'batch_size': batch_size,
        'lr': lr,
        'pos_weight': pos_weight_val,
        'balance_mu': balance_mu,
        'balance_sigma': balance_sigma
    }
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x, y in pbar:
            x[:, :, CHANNEL_MAP['jrc_occurrence'], :, :] = 0.0 # Kill JRC Leakage
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            autocast_context = torch.amp.autocast('cuda') if use_amp else torch.amp.autocast('cpu', enabled=False)
            with autocast_context:
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
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for x, y in pbar_val:
                x[:, :, CHANNEL_MAP['jrc_occurrence'], :, :] = 0.0
                x, y = x.to(device), y.to(device)
                
                autocast_context = torch.amp.autocast('cuda') if use_amp else torch.amp.autocast('cpu', enabled=False)
                with autocast_context:
                    logits, phys_pred, phys_inputs, hidden_states = model(x)
                    loss, loss_dict = criterion(logits, y, phys_pred, phys_inputs, hidden_states)
                
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_probs.append(probs.view(-1).cpu().numpy())
                all_targets.append(y.view(-1).cpu().numpy())
                
        val_loss /= len(val_loader)
        
        # Safeguard against empty validation set
        if len(all_probs) == 0:
            print("Validation dataset empty. Skipping validation metrics.")
            continue
            
        # Rigorous global epoch-level metric computation
        all_probs_cat = np.concatenate(all_probs)
        all_targets_cat = np.concatenate(all_targets)
        
        epoch_metrics = compute_flat_metrics(all_probs_cat, all_targets_cat)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val IoU: {epoch_metrics['iou']:.4f} | Val F1: {epoch_metrics['f1']:.4f}")
        print(f"Val Precision: {epoch_metrics['precision']:.4f} | Val Recall: {epoch_metrics['recall']:.4f}")
        print(f"Val ROC-AUC: {epoch_metrics['roc_auc']:.4f} | Val PR-AUC: {epoch_metrics['pr_auc']:.4f}")
        
        # Early Stopping on IoU
        avg_iou = epoch_metrics['iou']
        if avg_iou > best_val_iou:
            best_val_iou = avg_iou
            ckpt_path = f"{out_dir}/best_mfpit_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': avg_iou,
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path} (IoU: {avg_iou:.4f})")

if __name__ == "__main__":
    train()
