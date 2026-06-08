import os
import torch
import numpy as np
import json
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MFPITDataset
from MFPIT_Model import MFPIT
from metrics import compute_flat_metrics

def add_weight_noise(model, std=0.01, seed=42):
    """
    Applies Gaussian weight perturbation to all linear and convolutional parameters.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    perturbed_model = copy.deepcopy(model)
    with torch.no_grad():
        for param in perturbed_model.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * std
                param.add_(noise)
    return perturbed_model

def evaluate_perturbed_model(model, dataloader, device, threshold=0.030):
    model.eval()
    test_probs = []
    test_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            # Prevent JRC target leakage to stay scientifically consistent
            x = x.clone()
            x[:, :, 12, :, :] = 0.0
            x, y = x.to(device), y.to(device)
            
            logits, _, _, _ = model(x)
            probs = torch.sigmoid(logits)
            test_probs.append(probs.view(-1).cpu().numpy())
            test_targets.append(y.view(-1).cpu().numpy())
            
    test_probs_cat = np.concatenate(test_probs)
    test_targets_cat = np.concatenate(test_targets)
    
    metrics = compute_flat_metrics(test_probs_cat, test_targets_cat, threshold=threshold)
    return metrics

def main():
    print("=================================================================")
    print("--- Running Local Parameter Perturbation Sensitivity Analysis ---")
    print("=================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load original converged model
    model_path = "engineering_validation_results/best_mfpit_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Converged model checkpoint not found at {model_path}")
        return
        
    model = MFPIT().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load Test Set
    test_dataset = MFPITDataset(data_dir="../data/processed/tensors", split="test")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Noise levels to evaluate
    noise_levels = [0.001, 0.005, 0.010]
    n_trials = 3
    
    results = {}
    
    # 0. Control (no noise)
    print("Evaluating Control Model (No Noise)...")
    control_metrics = evaluate_perturbed_model(model, test_loader, device, threshold=0.030)
    print(f"Control -> IoU: {control_metrics['iou']:.4f} | F1: {control_metrics['f1']:.4f} | PR-AUC: {control_metrics['pr_auc']:.4f}")
    results['control'] = {
        'iou': float(control_metrics['iou']),
        'f1': float(control_metrics['f1']),
        'pr_auc': float(control_metrics['pr_auc'])
    }
    
    for sigma in noise_levels:
        print(f"\nEvaluating Noise level sigma = {sigma} ({n_trials} trials)...")
        ious = []
        f1s = []
        praucs = []
        
        for trial in range(n_trials):
            seed = 42 + trial
            perturbed = add_weight_noise(model, std=sigma, seed=seed)
            m = evaluate_perturbed_model(perturbed, test_loader, device, threshold=0.030)
            
            ious.append(m['iou'])
            f1s.append(m['f1'])
            praucs.append(m['pr_auc'])
            print(f"  Trial {trial+1} -> IoU: {m['iou']:.4f} | F1: {m['f1']:.4f} | PR-AUC: {m['pr_auc']:.4f}")
            
        results[f'sigma_{sigma}'] = {
            'iou': {
                'mean': float(np.mean(ious)),
                'std': float(np.std(ious)),
                'values': [float(v) for v in ious]
            },
            'f1': {
                'mean': float(np.mean(f1s)),
                'std': float(np.std(f1s)),
                'values': [float(v) for v in f1s]
            },
            'pr_auc': {
                'mean': float(np.mean(praucs)),
                'std': float(np.std(praucs)),
                'values': [float(v) for v in praucs]
            }
        }
        
    # Save results to JSON
    os.makedirs("engineering_validation_results", exist_ok=True)
    out_path = "engineering_validation_results/weight_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n=================================================================")
    print("--- SENSITIVITY SUMMARY ---")
    print("=================================================================")
    print(f"Control: IoU = {results['control']['iou']:.4f}, PR-AUC = {results['control']['pr_auc']:.4f}")
    for sigma in noise_levels:
        stats = results[f'sigma_{sigma}']
        print(f"Sigma = {sigma:<5}:")
        print(f"  - IoU:    {stats['iou']['mean']:.4f} \u00b1 {stats['iou']['std']:.4f}")
        print(f"  - PR-AUC: {stats['pr_auc']['mean']:.4f} \u00b1 {stats['pr_auc']['std']:.4f}")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
