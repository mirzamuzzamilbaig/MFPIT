import os
import sys
import torch
import numpy as np
import json
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import MFPITDataset
from metrics import compute_flat_metrics
from MFPIT_Model import MFPIT
from run_ablations import AblationDataset
from evaluate_all import two_stage_threshold_sweep

device = torch.device("cpu")

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
    
    # Wrap in Ablation Dataset (no_jrc mode to prevent JRC target leakage)
    val_dataset = AblationDataset(base_val, "no_jrc")
    test_dataset = AblationDataset(base_test, "no_jrc")
    
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
    
    test_metrics = compute_flat_metrics(test_probs_cat, test_targets_cat, threshold=best_t)
    
    return {
        'mode': mode,
        'threshold': best_t,
        'iou': test_metrics['iou'],
        'f1': test_metrics['f1'],
        'pr_auc': test_metrics['pr_auc']
    }

def main():
    print("=================================================================")
    print("--- Evaluating Physics Ablation: lambda_phys = 0 vs lambda_phys > 0 ---")
    print("=================================================================")
    
    results = []
    
    # 1. lambda = 0
    res_0 = evaluate_ablation_model("lambda_phys_0", "engineering_validation_results/ablation_lambda_0_model.pth")
    if res_0:
        results.append(res_0)
        
    # 2. lambda > 0 (represented by ablation_no_jrc_model.pth)
    res_plus = evaluate_ablation_model("lambda_phys_plus", "engineering_validation_results/ablation_no_jrc_model.pth")
    if res_plus:
        results.append(res_plus)
        
    if not results:
        print("No completed checkpoints found.")
        return
        
    print("\n\n=================================================================")
    print("--- PHYSICS ABLATION TEST SET BENCHMARKS SUMMARY ---")
    print("=================================================================")
    print(f"{'Ablation Mode':<18} | {'Optimal Threshold':<17} | {'Test IoU':<8} | {'Test F1':<7} | {'PR-AUC':<6}")
    print("-" * 68)
    
    for r in results:
        print(f"{r['mode']:<18} | {r['threshold']:<17.3f} | {r['iou']:<8.4f} | {r['f1']:<7.4f} | {r['pr_auc']:<6.4f}")
    print("=================================================================")
    
    # Save exact results to a JSON file
    out_path = "engineering_validation_results/physics_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"--> Saved physics ablation results to {out_path}")

if __name__ == "__main__":
    main()
