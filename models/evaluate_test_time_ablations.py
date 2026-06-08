import os
import sys
import torch
import numpy as np
import json
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import MFPITDataset
from metrics import compute_flat_metrics, compute_ece, compute_brier_score
from MFPIT_Model import MFPIT
from run_ablations import AblationDataset
from evaluate_all import two_stage_threshold_sweep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_test_time_ablation(mode, model, base_val, base_test):
    print(f"\nEvaluating Test-Time Masked Ablation Mode: {mode.upper()}...")
    
    # Wrap in Ablation Dataset
    val_dataset = AblationDataset(base_val, mode)
    test_dataset = AblationDataset(base_test, mode)
    
    # Data loaders (run in main thread for zero Windows spawn overhead)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # 1. Validation Threshold sweep
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
    print(f"  Optimal threshold chosen on Validation: {best_t:.3f} (Val IoU: {best_val_iou:.4f})")
    
    # 2. Test Split Evaluation
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
    
    # Save predicted probabilities for downstream validation or plotting
    out_dir = "engineering_validation_results"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/test_time_ablation_{mode}_test_probs.npy", test_probs_cat)
    np.save(f"{out_dir}/test_time_ablation_{mode}_test_targets.npy", test_targets_cat)
    
    test_metrics = compute_flat_metrics(test_probs_cat, test_targets_cat, threshold=best_t)
    ece = compute_ece(test_probs_cat, test_targets_cat)
    brier = compute_brier_score(test_probs_cat, test_targets_cat)
    
    return {
        'mode': mode,
        'threshold': best_t,
        'iou': test_metrics['iou'],
        'f1': test_metrics['f1'],
        'dice': test_metrics['dice'],
        'pr_auc': test_metrics['pr_auc'],
        'roc_auc': test_metrics['roc_auc'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'brier': brier,
        'ece': ece
    }

def run_test_time_ablations():
    print("=================================================================")
    print("--- Phase 5: Test-Time Sensitivity & Feature Masking Suite ---")
    print("=================================================================")
    
    ckpt_path = "engineering_validation_results/best_mfpit_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: Converged model checkpoint not found at {ckpt_path}.")
        return
        
    # Load converged model
    model = MFPIT().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load base datasets
    print("Loading base Validation and Test splits...")
    base_val = MFPITDataset(data_dir="../data/processed/tensors", split="val")
    base_test = MFPITDataset(data_dir="../data/processed/tensors", split="test")
    
    modes = ["static_full", "terrain_only", "no_jrc", "dynamic_only", "hydro_only", "eo_only"]
    results = []
    
    for mode in modes:
        res = evaluate_test_time_ablation(mode, model, base_val, base_test)
        results.append(res)
        
    # Print results summary
    print("\n\n=================================================================")
    print("--- TEST-TIME SENSITIVITY MASKING SUMMARY (ON CONVERGED WEIGHTS) ---")
    print("=================================================================")
    print(f"{'Ablation Mode':<15} | {'Threshold':<9} | {'Test IoU':<8} | {'Test F1':<7} | {'PR-AUC':<6} | {'ROC-AUC':<7} | {'ECE':<6}")
    print("-" * 75)
    
    for r in results:
        print(f"{r['mode']:<15} | {r['threshold']:<9.3f} | {r['iou']:<8.4f} | {r['f1']:<7.4f} | {r['pr_auc']:<6.4f} | {r['roc_auc']:<7.4f} | {r['ece']:<6.4f}")
    print("=================================================================")
    
    # Save results to JSON
    out_json = "engineering_validation_results/test_time_ablation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"--> Saved test-time ablation metrics to {out_json}")

if __name__ == "__main__":
    run_test_time_ablations()
