import os
import torch
import numpy as np
import pickle
import json
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, roc_auc_score, average_precision_score

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import MFPITDataset
from metrics import compute_flat_metrics, compute_ece, compute_brier_score
from MFPIT_Model import MFPIT
from baselines.unet_baseline import UNet
from baselines.cnn_lstm_baseline import CNNLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_dataset_for_eval(dataset):
    """
    Extracts ALL pixel-wise samples from ALL spatiotemporal patches
    for exact, unbiased classical model evaluation.
    """
    X_list = []
    y_list = []
    
    for i in range(len(dataset)):
        x, y = dataset[i]
        
        # JRC leakage prevention to match training
        x = x.clone()
        x[:, 12, :, :] = 0.0
        
        # Flatten spatial dims
        x_flat = x.reshape(12, 13, -1)
        y_flat = y.reshape(-1)
        
        # Flatten time (156 features per pixel)
        x_flat = x_flat.permute(2, 0, 1).reshape(-1, 12 * 13)
        
        X_list.append(x_flat.numpy())
        y_list.append(y_flat.numpy())
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"Extracted {len(X)} pixels from {len(dataset)} patches.")
    return X, y

def two_stage_threshold_sweep(val_probs, val_targets):
    """
    Finds the optimal threshold on the validation split using:
    Stage 1: Coarse sweep (0.05 to 0.95 with step 0.05)
    Stage 2: Fine sweep (around course winner with step 0.01)
    """
    # Stage 1: Coarse
    coarse_thresholds = np.arange(0.05, 1.0, 0.05)
    best_coarse_t = 0.5
    best_coarse_iou = -1.0
    
    for t in coarse_thresholds:
        preds = (val_probs > t).astype(np.float32)
        iou = jaccard_score(val_targets, preds, zero_division=0)
        if iou > best_coarse_iou:
            best_coarse_iou = iou
            best_coarse_t = t
            
    # Stage 2: Fine sweep around coarse winner (bounds of +/- 0.07, step 0.01)
    fine_thresholds = np.arange(max(0.01, best_coarse_t - 0.07), min(0.99, best_coarse_t + 0.07), 0.01)
    best_fine_t = best_coarse_t
    best_fine_iou = best_coarse_iou
    
    for t in fine_thresholds:
        preds = (val_probs > t).astype(np.float32)
        iou = jaccard_score(val_targets, preds, zero_division=0)
        if iou > best_fine_iou:
            best_fine_iou = iou
            best_fine_t = t
            
    return float(best_fine_t), float(best_fine_iou)

def evaluate_classical_model(model_name, pickle_path):
    print(f"\nEvaluating Classical Model: {model_name}...")
    if not os.path.exists(pickle_path):
        print(f"Error: {model_name} checkpoint not found at {pickle_path}.")
        return None
        
    with open(pickle_path, "rb") as f:
        model = pickle.load(f)
        
    # Load Validation Set to find optimal threshold
    print("Loading Validation Split for threshold sweep...")
    val_dataset = MFPITDataset(data_dir="../data/processed/tensors", split="val")
    
    # Flatten validation dataset deterministically
    X_val, y_val = flatten_dataset_for_eval(val_dataset)
    
    # Predict validation probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    
    # Perform Two-Stage Threshold Sweep
    best_t, best_val_iou = two_stage_threshold_sweep(val_probs, y_val)
    print(f"Optimal threshold chosen on Validation: {best_t:.3f} (Val IoU: {best_val_iou:.4f})")
    
    # Load Untouched Test Split
    print("Loading Untouched Test Split for final reporting...")
    test_dataset = MFPITDataset(data_dir="../data/processed/tensors", split="test")
    X_test, y_test = flatten_dataset_for_eval(test_dataset)
    
    # Predict test probabilities
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Save raw test predictions and targets
    np.save(f"engineering_validation_results/{model_name.lower().replace(' ', '_')}_test_probs.npy", test_probs)
    np.save(f"engineering_validation_results/{model_name.lower().replace(' ', '_')}_test_targets.npy", y_test)
    
    # Evaluate final test metrics using the optimized threshold
    test_metrics = compute_flat_metrics(test_probs, y_test, threshold=best_t)
    ece = compute_ece(test_probs, y_test)
    brier = compute_brier_score(test_probs, y_test)
    
    results = {
        'model': model_name,
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
    return results

def evaluate_deep_model(model_name, model_class, state_dict_path):
    print(f"\nEvaluating Deep Model: {model_name}...")
    if not os.path.exists(state_dict_path):
        print(f"Error: {model_name} checkpoint not found at {state_dict_path}.")
        return None
        
    model = model_class().to(device)
    checkpoint = torch.load(state_dict_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load Validation Split
    print("Loading Validation Split for threshold sweep...")
    val_dataset = MFPITDataset(data_dir="../data/processed/tensors", split="val")
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    val_probs = []
    val_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.clone()
            x[:, :, 12, :, :] = 0.0
            x, y = x.to(device), y.to(device)
            if model_name == "MFPIT":
                logits, _, _, _ = model(x)
            else:
                logits = model(x)
            probs = torch.sigmoid(logits)
            val_probs.append(probs.view(-1).cpu().numpy())
            val_targets.append(y.view(-1).cpu().numpy())
            
    val_probs_cat = np.concatenate(val_probs)
    val_targets_cat = np.concatenate(val_targets)
    
    # Perform Two-Stage Threshold Sweep
    best_t, best_val_iou = two_stage_threshold_sweep(val_probs_cat, val_targets_cat)
    print(f"Optimal threshold chosen on Validation: {best_t:.3f} (Val IoU: {best_val_iou:.4f})")
    
    # Load Untouched Test Split
    print("Loading Untouched Test Split for final reporting...")
    test_dataset = MFPITDataset(data_dir="../data/processed/tensors", split="test")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    test_probs = []
    test_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.clone()
            x[:, :, 12, :, :] = 0.0
            x, y = x.to(device), y.to(device)
            if model_name == "MFPIT":
                logits, _, _, _ = model(x)
            else:
                logits = model(x)
            probs = torch.sigmoid(logits)
            test_probs.append(probs.view(-1).cpu().numpy())
            test_targets.append(y.view(-1).cpu().numpy())
            
    test_probs_cat = np.concatenate(test_probs)
    test_targets_cat = np.concatenate(test_targets)
    
    # Save raw test probabilities for reliability diagrams / PR curves
    np.save(f"engineering_validation_results/{model_name.lower()}_test_probs.npy", test_probs_cat)
    np.save(f"engineering_validation_results/{model_name.lower()}_test_targets.npy", test_targets_cat)
    
    # Evaluate final test metrics using the optimized threshold
    test_metrics = compute_flat_metrics(test_probs_cat, test_targets_cat, threshold=best_t)
    ece = compute_ece(test_probs_cat, test_targets_cat)
    brier = compute_brier_score(test_probs_cat, test_targets_cat)
    
    results = {
        'model': model_name,
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
    return results

def run_full_evaluation():
    print("=================================================================")
    print("--- Phase 2/3: Publication-Grade Evaluation & Threshold Calibration Suite ---")
    print("=================================================================")
    
    results_list = []
    
    # 1. Random Forest
    rf_res = evaluate_classical_model("Random Forest", "engineering_validation_results/rf_baseline.pkl")
    if rf_res: results_list.append(rf_res)
    
    # 2. GBDT
    gbdt_res = evaluate_classical_model("GBDT", "engineering_validation_results/gbdt_baseline.pkl")
    if gbdt_res: results_list.append(gbdt_res)
    
    # 3. U-Net
    unet_res = evaluate_deep_model("U-Net", UNet, "engineering_validation_results/unet_baseline.pth")
    if unet_res: results_list.append(unet_res)
    
    # 4. CNN-LSTM
    cnn_lstm_res = evaluate_deep_model("CNN-LSTM", CNNLSTM, "engineering_validation_results/cnn_lstm_baseline.pth")
    if cnn_lstm_res: results_list.append(cnn_lstm_res)
    
    # 5. MFPIT
    mfpit_res = evaluate_deep_model("MFPIT", MFPIT, "engineering_validation_results/best_mfpit_model.pth")
    if mfpit_res: results_list.append(mfpit_res)
    
    # Compile Results
    print("\n\n=================================================================")
    print("--- FINAL UNTOUCHED TEST SPLIT BENCHMARK RESULTS ---")
    print("=================================================================")
    print(f"{'Model':<20} | {'Threshold':<9} | {'IoU':<6} | {'F1-Score':<8} | {'PR-AUC':<6} | {'ROC-AUC':<7} | {'Brier':<6} | {'ECE':<6}")
    print("-" * 92)
    
    for r in results_list:
        print(f"{r['model']:<20} | {r['threshold']:<9.3f} | {r['iou']:<6.4f} | {r['f1']:<8.4f} | {r['pr_auc']:<6.4f} | {r['roc_auc']:<7.4f} | {r['brier']:<6.4f} | {r['ece']:<6.4f}")
        
    print("=================================================================")
    
    # Reviewer-grade check: Save exact metrics to JSON to ensure absolute bookkeeping cleanliness
    out_json = "engineering_validation_results/final_full_model_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results_list, f, indent=4)
    print(f"--> Saved exact full model metrics table to {out_json}")

if __name__ == "__main__":
    run_full_evaluation()
