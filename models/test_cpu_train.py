import sys
import os
import time
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import MFPITDataset
from MFPIT_Model import MFPIT
from losses import MFPITLoss

def main():
    device = torch.device("cpu")
    print("Loading dataset...")
    train_dataset = MFPITDataset(data_dir="../data/processed/tensors", split="train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print("Initializing model...")
    model = MFPIT().to(device)
    criterion = MFPITLoss(lambda_phys=0.0, lambda_temp=0.1, pos_weight=5.0, balance_mu=0.0, balance_sigma=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("Starting CPU batch timing test...")
    start_time = time.time()
    
    for i, (x, y) in enumerate(train_loader):
        if i >= 5:
            break
        batch_start = time.time()
        x = x.clone()
        x[:, :, 12, :, :] = 0.0 # prevent leakage
        
        logits, phys_pred, phys_inputs, hidden_states = model(x)
        loss, _ = criterion(logits, y, phys_pred, phys_inputs, hidden_states)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_end = time.time()
        print(f"Batch {i+1}/5 completed in {batch_end - batch_start:.2f} seconds.")
        
    total_time = time.time() - start_time
    print(f"Total time for 5 batches: {total_time:.2f} seconds.")
    print(f"Estimated time per batch: {total_time/5:.2f} seconds.")
    print(f"Estimated time for 1 full epoch (465 batches): {total_time/5 * 465 / 60:.2f} minutes.")
    print(f"Estimated time for 25 epochs: {total_time/5 * 465 * 25 / 3600:.2f} hours.")

if __name__ == "__main__":
    main()
