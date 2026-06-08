import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import MFPITDataset
from metrics import compute_flat_metrics
from MFPIT_Model import MFPIT
from run_ablations import AblationDataset
from evaluate_all import two_stage_threshold_sweep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_ablation_model(mode, ckpt_path):
    print(f"\nEvaluating Ablated Model: {mode.upper()}...")
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found at {ckpt_path}.")
        return None
        
    model = MFPIT().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load base dataset
    base_val = MFPITDataset(data_dir="../data/processed/tensors", split="val")
    base_test = MFPITDataset(data_dir="../data/processed/tensors", split="test")
    
    # Wrap in Ablation Dataset
    val_dataset = AblationDataset(base_val, mode)
    test_dataset = AblationDataset(base_test, mode)
    
    # Data loaders
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Validation Sweep
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, _, _, _ = model(x)
            probs = torch.sigmoid(logits)
            val_probs.append(probs.view(-1).cpu().numpy())
            val_targets.append(y.view(-1).cpu().numpy())
            
    val_probs_cat = np.concatenate(val_probs)
    val_targets_cat = np.concatenate(val_targets)
    
    best_t, best_val_iou = two_stage_threshold_sweep(val_probs_cat, val_targets_cat)
    print(f"Optimal threshold chosen on Validation: {best_t:.3f} (Val IoU: {best_val_iou:.4f})")
    
    # Test Evaluation
    test_probs = []
    test_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _, _, _ = model(x)
            probs = torch.sigmoid(logits)
            test_probs.append(probs.view(-1).cpu().numpy())
            test_targets.append(y.view(-1).cpu().numpy())
            
    test_probs_cat = np.concatenate(test_probs)
    test_targets_cat = np.concatenate(test_targets)
    
    # Save predicted probabilities for downstream analysis
    np.save(f"engineering_validation_results/ablation_{mode}_test_probs.npy", test_probs_cat)
    np.save(f"engineering_validation_results/ablation_{mode}_test_targets.npy", test_targets_cat)
    
    test_metrics = compute_flat_metrics(test_probs_cat, test_targets_cat, threshold=best_t)
    
    return {
        'mode': mode,
        'threshold': best_t,
        'iou': test_metrics['iou'],
        'f1': test_metrics['f1'],
        'pr_auc': test_metrics['pr_auc']
    }

def run_ablation_evaluations():
    print("=================================================================")
    print("--- Phase 4: Publication-Grade Ablation Evaluation Suite ---")
    print("=================================================================")
    
    modes = ["static_full", "terrain_only", "no_jrc", "dynamic_only", "hydro_only", "eo_only"]
    results = []
    
    for mode in modes:
        ckpt_path = f"engineering_validation_results/ablation_{mode}_model.pth"
        res = evaluate_ablation_model(mode, ckpt_path)
        if res:
            results.append(res)
            
    if not results:
        print("No completed ablation checkpoints found. Run 'python run_ablations.py --ablation <mode>' first.")
        return
        
    print("\n\n=================================================================")
    print("--- ABLATION TEST SET BENCHMARKS SUMMARY ---")
    print("=================================================================")
    print(f"{'Ablation Mode':<15} | {'Optimal Threshold':<17} | {'Test IoU':<8} | {'Test F1':<7} | {'PR-AUC':<6}")
    print("-" * 65)
    
    for r in results:
        print(f"{r['mode']:<15} | {r['threshold']:<17.3f} | {r['iou']:<8.4f} | {r['f1']:<7.4f} | {r['pr_auc']:<6.4f}")
    print("=================================================================")

if __name__ == "__main__":
    run_ablation_evaluations()
