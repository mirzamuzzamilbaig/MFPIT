import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import sys

# Append parent dir to path to import dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import MFPITDataset
from losses import DiceLoss

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=156, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Input: [B, 12, 13, 64, 64] -> Flatten time and channels: [B, 156, 64, 64]
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return logits

def train_unet_baseline(epochs=2):
    print("\n--- Training U-Net Baseline ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = MFPITDataset(data_dir="../../data/processed/tensors", split="train")
    val_dataset = MFPITDataset(data_dir="../../data/processed/tensors", split="val")
    
    if len(train_dataset) == 0:
        print("Data extraction incomplete. Waiting for real tensors.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Import validation metrics
    from metrics import compute_flat_metrics
    
    model = UNet(in_channels=156, out_channels=1).to(device)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_iou = -1.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"U-Net Epoch {epoch+1}/{epochs} [Train]")
        
        for x, y in pbar:
            # JRC leakage prevention
            x[:, :, 12, :, :] = 0.0
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x[:, :, 12, :, :] = 0.0
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = bce(logits, y) + dice(logits, y)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_probs.append(probs.view(-1).cpu().numpy())
                all_targets.append(y.view(-1).cpu().numpy())
                
        val_loss /= len(val_loader)
        
        if len(all_probs) > 0:
            all_probs_cat = np.concatenate(all_probs)
            all_targets_cat = np.concatenate(all_targets)
            epoch_metrics = compute_flat_metrics(all_probs_cat, all_targets_cat)
            
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Val IoU: {epoch_metrics['iou']:.4f} | Val F1: {epoch_metrics['f1']:.4f}")
            print(f"  Val Precision: {epoch_metrics['precision']:.4f} | Val Recall: {epoch_metrics['recall']:.4f}")
            print(f"  Val ROC-AUC: {epoch_metrics['roc_auc']:.4f} | Val PR-AUC: {epoch_metrics['pr_auc']:.4f}")
            
            avg_iou = epoch_metrics['iou']
            if avg_iou > best_val_iou:
                best_val_iou = avg_iou
                os.makedirs("../engineering_validation_results", exist_ok=True)
                torch.save(model.state_dict(), "../engineering_validation_results/unet_baseline.pth")
                print(f"  Saved best U-Net checkpoint (Val IoU: {best_val_iou:.4f})")
        else:
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} (No validation data)")
            
    print(f"U-Net Baseline training complete. Best Val IoU: {best_val_iou:.4f}")
    
    # Final untouched test split evaluation to eliminate validation selection leakage bias
    print("\nEvaluating Best U-Net Model on Independent Untouched Test Set...")
    best_path = "../engineering_validation_results/unet_baseline.pth"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        
    test_dataset = MFPITDataset(data_dir="../../data/processed/tensors", split="test")
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        model.eval()
        
        test_probs = []
        test_targets = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x[:, :, 12, :, :] = 0.0
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)
                test_probs.append(probs.view(-1).cpu().numpy())
                test_targets.append(y.view(-1).cpu().numpy())
                
        test_probs_cat = np.concatenate(test_probs)
        test_targets_cat = np.concatenate(test_targets)
        test_metrics = compute_flat_metrics(test_probs_cat, test_targets_cat)
        
        print(f"\n[U-Net Final Untouched Test Results]")
        print(f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")
        print(f"IoU: {test_metrics['iou']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}")
    else:
        print("Test dataset not found.")

if __name__ == "__main__":
    train_unet_baseline(epochs=2)
