import torch
import numpy as np
from MFPIT_Model import MFPIT
from dataset import MFPITDataset
import torch.nn as nn

def enable_mc_dropout(model):
    """
    Enables MC Dropout consistently by explicitly setting all
    Dropout and Dropout2d modules to train mode during inference.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

def run_mc_dropout_inference(model, x, num_passes=30):
    """
    Executes MC Dropout inference protocol.
    x: [B, 12, 13, H, W]
    Returns: mean_prediction [B, 1, H, W], variance_map [B, 1, H, W]
    """
    enable_mc_dropout(model)
    
    device = next(model.parameters()).device
    x = x.to(device)
    
    predictions = []
    
    with torch.no_grad():
        for i in range(num_passes):
            # Only need logits for inference map
            logits, _, _, _ = model(x)
            probs = torch.sigmoid(logits)
            predictions.append(probs)
            
    # Stack along new dimension: [passes, B, 1, H, W]
    stacked_preds = torch.stack(predictions)
    
    mean_prediction = torch.mean(stacked_preds, dim=0)
    variance_map = torch.var(stacked_preds, dim=0)
    
    return mean_prediction, variance_map

if __name__ == "__main__":
    print("Testing MC Dropout Inference Protocol...")
    model = MFPIT()
    x = torch.randn(1, 12, 13, 64, 64)
    
    mean_pred, var_map = run_mc_dropout_inference(model, x, num_passes=30)
    
    print(f"Mean shape: {mean_pred.shape}")
    print(f"Variance shape: {var_map.shape}")
    print(f"Max Variance: {var_map.max().item():.4f}")
