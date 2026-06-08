import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import numpy as np

def compute_metrics(logits, targets, threshold=0.5):
    """
    logits: [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    probs = torch.sigmoid(logits)
    
    # Flatten and convert to numpy for clean, standardized evaluation
    probs_flat = probs.view(-1).cpu().numpy()
    targets_flat = targets.view(-1).cpu().numpy()
    
    return compute_flat_metrics(probs_flat, targets_flat, threshold)

def compute_flat_metrics(probs_flat, targets_flat, threshold=0.5):
    """
    Computes rigorous global epoch-level metrics from concatenated flat arrays.
    Prevents batch-mean bias.
    """
    preds_flat = (probs_flat > threshold).astype(np.float32)
    
    tp = np.sum((preds_flat == 1.0) & (targets_flat == 1.0))
    tn = np.sum((preds_flat == 0.0) & (targets_flat == 0.0))
    fp = np.sum((preds_flat == 1.0) & (targets_flat == 0.0))
    fn = np.sum((preds_flat == 0.0) & (targets_flat == 1.0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    if len(np.unique(targets_flat)) > 1:
        roc_auc = roc_auc_score(targets_flat, probs_flat)
        pr_auc = average_precision_score(targets_flat, probs_flat)
    else:
        roc_auc = float('nan')
        pr_auc = float('nan')
        
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'dice': float(dice),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

def compute_ece(probs, targets, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for pixel-level binary classification.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Get elements in this bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(targets[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)

def compute_brier_score(probs, targets):
    """
    Computes Brier Score (Mean Squared Error between predicted probabilities and labels).
    """
    return float(np.mean((probs - targets) ** 2))

