import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys

# Append parent dir to path to import dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import MFPITDataset
from losses import DiceLoss

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=13, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)

class CNNDecoder(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim * 8 * 8),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 128, 2, stride=2), # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2), # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), # 32 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(x.size(0), -1, 8, 8)
        return self.decoder(x)

class CNNLSTM(nn.Module):
    def __init__(self, in_channels=13, hidden_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, out_dim=hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.decoder = CNNDecoder(in_dim=hidden_dim)

    def forward(self, x):
        # Input: [B, T=12, C=13, H, W]
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        # Encode each month spatially: [B*12, hidden_dim]
        embs = self.encoder(x_flat)
        embs = embs.view(B, T, -1) # [B, 12, hidden_dim]
        
        # Temporal transition using LSTM
        lstm_out, (hn, cn) = self.lstm(embs) # lstm_out: [B, 12, hidden_dim]
        
        # Take final temporal state
        final_state = lstm_out[:, -1, :] # [B, hidden_dim]
        
        # Decode to 2D prediction map: [B, 1, H, W]
        logits = self.decoder(final_state)
        return logits

def train_cnn_lstm_baseline(epochs=2):
    print("\n--- Training CNN-LSTM Baseline ---")
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
    
    model = CNNLSTM().to(device)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_iou = -1.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"CNN-LSTM Epoch {epoch+1}/{epochs} [Train]")
        
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
                torch.save(model.state_dict(), "../engineering_validation_results/cnn_lstm_baseline.pth")
                print(f"  Saved best CNN-LSTM checkpoint (Val IoU: {best_val_iou:.4f})")
        else:
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} (No validation data)")
            
    print(f"CNN-LSTM Baseline training complete. Best Val IoU: {best_val_iou:.4f}")
    
    # Final untouched test split evaluation to eliminate validation selection leakage bias
    print("\nEvaluating Best CNN-LSTM Model on Independent Untouched Test Set...")
    best_path = "../engineering_validation_results/cnn_lstm_baseline.pth"
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
        
        print(f"\n[CNN-LSTM Final Untouched Test Results]")
        print(f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")
        print(f"IoU: {test_metrics['iou']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}")
    else:
        print("Test dataset not found.")

if __name__ == "__main__":
    train_cnn_lstm_baseline(epochs=2)
