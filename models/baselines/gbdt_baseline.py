import os
import torch
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, jaccard_score
from tqdm import tqdm
import pickle
import sys

# Append parent dir to path to import dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import MFPITDataset

def flatten_dataset_for_gbdt(dataset, max_samples=50000):
    """
    Extracts random pixel-wise samples from the spatiotemporal patches
    to train a classical model like GBDT.
    """
    print("Flattening dataset into pixel-wise samples for classical ML...")
    
    X_list = []
    y_list = []
    
    # We sample a subset of patches to fit in RAM
    num_patches = min(100, len(dataset))
    
    # Reproducible random seed for sampling to eliminate temporal/spatial bias
    np.random.seed(42)
    indices = np.random.choice(len(dataset), num_patches, replace=False)
    
    for idx in tqdm(indices, desc="Extracting pixels"):
        x, y = dataset[int(idx)] # x: [12, 13, 64, 64], y: [1, 64, 64]
        
        # JRC leakage prevention
        x[:, 12, :, :] = 0.0
        
        # Flatten spatial dims: [12, 13, 4096]
        x_flat = x.reshape(12, 13, -1)
        y_flat = y.reshape(-1)
        
        # Flatten time (12 * 13 = 156 features per pixel)
        x_flat = x_flat.permute(2, 0, 1).reshape(-1, 12 * 13) # [4096, 156]
        
        X_list.append(x_flat.numpy())
        y_list.append(y_flat.numpy())
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Subsample to avoid memory crash
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]
        
    print(f"Extracted {len(X)} pixels with {X.shape[1]} features.")
    print(f"Label prevalence: Water={(y==1).mean()*100:.2f}% | Non-Water={(y==0).mean()*100:.2f}%")
    return X, y

def train_gbdt_baseline():
    print("\n--- Training Gradient Boosted Decision Tree (GBDT) Baseline ---")
    train_dataset = MFPITDataset(data_dir="../../data/processed/tensors", split="train")
    val_dataset = MFPITDataset(data_dir="../../data/processed/tensors", split="val")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Data extraction incomplete. Waiting for real tensors.")
        return
        
    X_train, y_train = flatten_dataset_for_gbdt(train_dataset, max_samples=100000)
    X_val, y_val = flatten_dataset_for_gbdt(val_dataset, max_samples=50000)
    
    # Validation grid search to ensure equal selection rigor with deep baselines
    param_grid = [
        {'max_iter': 50, 'max_depth': 5},
        {'max_iter': 100, 'max_depth': 5},
        {'max_iter': 50, 'max_depth': 10},
        {'max_iter': 100, 'max_depth': 10}
    ]
    
    best_gbdt = None
    best_iou = -1.0
    best_params = None
    
    for params in param_grid:
        print(f"\nEvaluating GBDT with params: {params}...")
        gbdt = HistGradientBoostingClassifier(
            max_iter=params['max_iter'],
            max_depth=params['max_depth'],
            random_state=42,
            verbose=0
        )
        gbdt.fit(X_train, y_train)
        
        y_pred = gbdt.predict(X_val)
        iou = jaccard_score(y_val, y_pred, zero_division=0)
        print(f"  Validation IoU: {iou:.4f}")
        
        if iou > best_iou:
            best_iou = iou
            best_gbdt = gbdt
            best_params = params
            
    print(f"\nBest GBDT hyperparameters selected: {best_params} (Val IoU: {best_iou:.4f})")
    
    # Final evaluation on Val using best model
    y_pred = best_gbdt.predict(X_val)
    y_probs = best_gbdt.predict_proba(X_val)[:, 1]
    
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_probs) if len(np.unique(y_val)) > 1 else float('nan')
    
    print("\n[GBDT Final Validation Results]")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"IoU: {best_iou:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # Final untouched test split evaluation to eliminate validation selection leakage bias
    print("\nEvaluating Best Model on Independent Untouched Test Set...")
    test_dataset = MFPITDataset(data_dir="../../data/processed/tensors", split="test")
    X_test, y_test = flatten_dataset_for_gbdt(test_dataset, max_samples=50000)
    
    y_pred_test = best_gbdt.predict(X_test)
    y_probs_test = best_gbdt.predict_proba(X_test)[:, 1]
    
    precision_test = precision_score(y_test, y_pred_test, zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)
    iou_test = jaccard_score(y_test, y_pred_test, zero_division=0)
    roc_auc_test = roc_auc_score(y_test, y_probs_test) if len(np.unique(y_test)) > 1 else float('nan')
    
    print(f"\n[GBDT Final Untouched Test Results]")
    print(f"Precision: {precision_test:.4f} | Recall: {recall_test:.4f} | F1: {f1_test:.4f}")
    print(f"IoU: {iou_test:.4f} | ROC-AUC: {roc_auc_test:.4f}")
    
    os.makedirs("../engineering_validation_results", exist_ok=True)
    with open("../engineering_validation_results/gbdt_baseline.pkl", "wb") as f:
        pickle.dump(best_gbdt, f)
        
    print("GBDT Baseline saved.")

if __name__ == "__main__":
    train_gbdt_baseline()
