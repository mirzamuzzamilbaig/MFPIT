import os
import sys
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import average_precision_score

def compute_fast_metrics(probs_flat, targets_flat, threshold=0.5):
    preds_flat = (probs_flat > threshold).astype(np.float32)
    
    tp = np.sum((preds_flat == 1.0) & (targets_flat == 1.0))
    tn = np.sum((preds_flat == 0.0) & (targets_flat == 0.0))
    fp = np.sum((preds_flat == 1.0) & (targets_flat == 0.0))
    fn = np.sum((preds_flat == 0.0) & (targets_flat == 1.0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Fast PR-AUC using sklearn on downsampled pixels
    if len(np.unique(targets_flat)) > 1:
        pr_auc = average_precision_score(targets_flat, probs_flat)
    else:
        pr_auc = float('nan')
        
    return {
        'iou': float(iou),
        'f1': float(f1),
        'pr_auc': float(pr_auc)
    }

def run_spatial_bootstrap(model_name, probs_path, targets_path, threshold, B=500, seed=42):
    print(f"Running Spatial Block Bootstrap for {model_name}...")
    if not os.path.exists(probs_path) or not os.path.exists(targets_path):
        print(f"Error: {probs_path} or {targets_path} not found.")
        return None
        
    probs = np.load(probs_path)
    targets = np.load(targets_path)
    
    # Shape is (802816,) -> 196 patches of 64x64
    n_patches = len(probs) // 4096
    
    # Reshape to patches
    probs_patches = probs[:n_patches * 4096].reshape(n_patches, 64, 64)
    targets_patches = targets[:n_patches * 4096].reshape(n_patches, 64, 64)
    
    # Spatial downsampling to speed up PR-AUC sort from 800k to 50k pixels
    # Taking every 4th pixel in H and W retains spatial structure and flow boundaries perfectly
    probs_down = probs_patches[:, ::4, ::4].reshape(n_patches, -1)
    targets_down = targets_patches[:, ::4, ::4].reshape(n_patches, -1)
    
    np.random.seed(seed)
    
    boot_ious = []
    boot_f1s = []
    boot_praucs = []
    
    for _ in tqdm(range(B), desc=f"Bootstrap {model_name}"):
        # Sample patch indices with replacement
        boot_idx = np.random.choice(n_patches, n_patches, replace=True)
        
        boot_probs = probs_down[boot_idx].reshape(-1)
        boot_targets = targets_down[boot_idx].reshape(-1)
        
        metrics = compute_fast_metrics(boot_probs, boot_targets, threshold=threshold)
        
        boot_ious.append(metrics['iou'])
        boot_f1s.append(metrics['f1'])
        boot_praucs.append(metrics['pr_auc'])
        
    boot_ious = np.array(boot_ious)
    boot_f1s = np.array(boot_f1s)
    boot_praucs = np.array(boot_praucs)
    
    results = {
        'iou': {
            'mean': float(np.mean(boot_ious)),
            'std': float(np.std(boot_ious)),
            'ci_lower': float(np.percentile(boot_ious, 2.5)),
            'ci_upper': float(np.percentile(boot_ious, 97.5))
        },
        'f1': {
            'mean': float(np.mean(boot_f1s)),
            'std': float(np.std(boot_f1s)),
            'ci_lower': float(np.percentile(boot_f1s, 2.5)),
            'ci_upper': float(np.percentile(boot_f1s, 97.5))
        },
        'pr_auc': {
            'mean': float(np.mean(boot_praucs)),
            'std': float(np.std(boot_praucs)),
            'ci_lower': float(np.percentile(boot_praucs, 2.5)),
            'ci_upper': float(np.percentile(boot_praucs, 97.5))
        }
    }
    
    return results

def main():
    models_config = {
        "Random Forest": {
            "probs": "engineering_validation_results/random_forest_test_probs.npy",
            "targets": "engineering_validation_results/random_forest_test_targets.npy",
            "threshold": 0.250
        },
        "GBDT": {
            "probs": "engineering_validation_results/gbdt_test_probs.npy",
            "targets": "engineering_validation_results/gbdt_test_targets.npy",
            "threshold": 0.110
        },
        "U-Net": {
            "probs": "engineering_validation_results/u-net_test_probs.npy",
            "targets": "engineering_validation_results/u-net_test_targets.npy",
            "threshold": 0.100
        },
        "CNN-LSTM": {
            "probs": "engineering_validation_results/cnn-lstm_test_probs.npy",
            "targets": "engineering_validation_results/cnn-lstm_test_targets.npy",
            "threshold": 0.490
        },
        "MFPIT": {
            "probs": "engineering_validation_results/mfpit_test_probs.npy",
            "targets": "engineering_validation_results/mfpit_test_targets.npy",
            "threshold": 0.030
        },
        "Ablation Terrain-only": {
            "probs": "engineering_validation_results/ablation_terrain_only_test_probs.npy",
            "targets": "engineering_validation_results/ablation_terrain_only_test_targets.npy",
            "threshold": 0.640
        },
        "Ablation No-JRC": {
            "probs": "engineering_validation_results/ablation_no_jrc_test_probs.npy",
            "targets": "engineering_validation_results/ablation_no_jrc_test_targets.npy",
            "threshold": 0.330
        },
        "Ablation Dynamic-only": {
            "probs": "engineering_validation_results/ablation_dynamic_only_test_probs.npy",
            "targets": "engineering_validation_results/ablation_dynamic_only_test_targets.npy",
            "threshold": 0.080
        }
    }
    
    all_results = {}
    
    for model_name, cfg in models_config.items():
        res = run_spatial_bootstrap(
            model_name, 
            cfg['probs'], 
            cfg['targets'], 
            cfg['threshold'], 
            B=200, 
            seed=42
        )
        if res:
            all_results[model_name] = res
            
    # Save to JSON
    os.makedirs("engineering_validation_results", exist_ok=True)
    with open("engineering_validation_results/bootstrap_ci_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print("\n\n========================================================")
    print("      SPATIAL BLOCK BOOTSTRAP 95% CONFIDENCE INTERVALS  ")
    print("========================================================")
    for model_name, metrics in all_results.items():
        print(f"\nModel: {model_name}")
        for metric_name, stats in metrics.items():
            print(f"  - {metric_name.upper():<6}: {stats['mean']:.4f} ± {stats['std']:.4f} (95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])")
    print("========================================================\n")

if __name__ == "__main__":
    main()
