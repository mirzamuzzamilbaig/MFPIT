import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataset import MFPITDataset
from config import CHANNEL_MAP

def compute_physics_stats(data_dir="../data/processed/tensors", output_file="physics_stats.json"):
    print("Computing Physics Normalization Statistics over Training Set...")
    
    # We only use the training set to prevent leakage
    dataset = MFPITDataset(data_dir=data_dir, split="train", cache_size=1)
    
    if len(dataset) == 0:
        print("No training data found. Cannot compute stats.")
        return
        
    balances = []
    
    # Optional optimization: sample a subset of 50k patches for speed with reproducibility seed
    np.random.seed(42)
    num_samples = min(50000, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False) if len(dataset) > num_samples else range(len(dataset))
    
    for i in tqdm(indices, desc="Scanning Patches"):
        x, _ = dataset[i] # x: [12, 13, 64, 64]
        
        # P, ET, R mapping via config
        p = x[:, CHANNEL_MAP['precip'], :, :].sum(dim=0).mean().item()
        et = x[:, CHANNEL_MAP['et'], :, :].sum(dim=0).mean().item()
        r = x[:, CHANNEL_MAP['runoff'], :, :].sum(dim=0).mean().item()
        
        balance = p - et - r
        balances.append(balance)
        
    mu = np.mean(balances)
    sigma = np.std(balances) + 1e-8
    
    stats = {
        "balance_mu": float(mu),
        "balance_sigma": float(sigma)
    }
    
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"Saved frozen physics stats to {output_file}:")
    print(f"  mu: {mu:.4f}")
    print(f"  sigma: {sigma:.4f}")

if __name__ == "__main__":
    compute_physics_stats()
