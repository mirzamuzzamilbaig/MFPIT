import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CHANNEL_MAP

class SpatialEncoder(nn.Module):
    """
    Shared CNN Encoder per month.
    Input: [B, C, H, W] where C=13
    Output: [B, 256, H', W']
    """
    def __init__(self, in_channels=13):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 64x64 -> 32x32
            nn.Dropout2d(p=0.1) # Light dropout early
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 32x32 -> 16x16
            nn.Dropout2d(p=0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 16x16 -> 8x8
            nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return e3, e2, e1 

class TemporalTransformer(nn.Module):
    """
    Transformer temporal encoder.
    Tokenization explicitly documented as Case A: Global Monthly Tokens.
    Input: [B, 12, 256] -> one 256-d vector per month.
    """
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        # PyTorch TransformerEncoderLayer inherently contains dropout (default 0.1)
        # We explicitly configure it for MC Dropout depth
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, 12, d_model)) # 12 months
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x: [B, T=12, d_model]
        x = x + self.pos_emb
        x = self.dropout(x)
        out = self.transformer(x)
        return out

class PhysicsBranch(nn.Module):
    """
    Explicit Proxy Hydrological Branch.
    Extracts P, ET, runoff, soil from the input based on CHANNEL_MAP.
    """
    def __init__(self, d_model=256, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 64 -> 32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 32 -> 16
            nn.Conv2d(128, out_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 16 -> 8
        )
        
        self.storage_proxy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_dim, 1) # Predicts single proxy value for storage change
        )

    def forward(self, x_physics):
        # x_physics: [B, 4, H, W]
        emb = self.proj(x_physics) # [B, out_dim, 8, 8]
        storage_pred = self.storage_proxy_head(emb) # [B, 1]
        return emb, storage_pred

class FusionDecoder(nn.Module):
    """
    Decoder upsampling: 256 -> 128 -> 64 -> 1
    Includes Attention Fusion and MC Dropout for uncertainty.
    """
    def __init__(self, in_channels=256):
        super().__init__()
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(in_channels, 128, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # MC Dropout added for uncertainty quantification
        self.mc_dropout = nn.Dropout2d(p=0.2)
        
        self.head = nn.Conv2d(32, 1, 1)
        
    def forward(self, x):
        x = self.fusion_proj(x)
        
        x = self.up1(x)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = self.conv3(x)
        
        # Apply dropout even during inference if MC Dropout is enabled
        x = self.mc_dropout(x)
        logits = self.head(x)
        
        return logits

class MFPIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels=13)
        self.temporal_encoder = TemporalTransformer(d_model=256, nhead=8, num_layers=4)
        self.physics_branch = PhysicsBranch(d_model=256, out_dim=256)
        self.decoder = FusionDecoder(in_channels=256)

    def forward(self, x):
        """
        Input x: [B, T=12, C=13, H, W]
        """
        B, T, C, H, W = x.shape
        
        # 1. Spatial Encoding per month
        x_flat = x.view(B * T, C, H, W)
        e3, _, _ = self.spatial_encoder(x_flat) # [B*T, 256, 8, 8]
        
        _, C_out, H_out, W_out = e3.shape
        e3_bt = e3.view(B, T, C_out, H_out, W_out)
        
        # 2. Temporal Encoding (Case A: Global Monthly Tokens)
        e3_pool = F.adaptive_avg_pool2d(e3, 1).view(B, T, C_out) # [B, 12, 256]
        
        temporal_features = self.temporal_encoder(e3_pool) # [B, 12, 256]
        hidden_states = temporal_features # Save for temporal consistency loss
        
        t_agg = temporal_features.mean(dim=1) # [B, 256]
        t_emb_spatial = t_agg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H_out, W_out) # [B, 256, 8, 8]
        
        # 3. Physics Branch
        phys_channels = [CHANNEL_MAP['precip'], CHANNEL_MAP['et'], CHANNEL_MAP['soil'], CHANNEL_MAP['runoff']]
        phys_input_channels = x[:, :, phys_channels, :, :].mean(dim=1) # [B, 4, H, W]
        
        p_ch = x[:, :, CHANNEL_MAP['precip'], :, :].sum(dim=1) # [B, H, W]
        et_ch = x[:, :, CHANNEL_MAP['et'], :, :].sum(dim=1) # [B, H, W]
        r_ch = x[:, :, CHANNEL_MAP['runoff'], :, :].sum(dim=1) # [B, H, W]
        soil_ch = x[:, :, CHANNEL_MAP['soil'], :, :].sum(dim=1) # [B, H, W]
        
        phys_emb, phys_pred = self.physics_branch(phys_input_channels) # phys_pred: [B, 1]
        
        # 4. Fusion
        s_emb = e3_bt.mean(dim=1) # [B, 256, 8, 8]
        fused = torch.cat([s_emb, t_emb_spatial, phys_emb], dim=1) # [B, 768, 8, 8]
        
        # 5. Decoder
        logits = self.decoder(fused) # [B, 1, 64, 64]
        
        phys_inputs = {
            'P': p_ch.mean(dim=[-1, -2]),   # Global/Mean values [B]
            'ET': et_ch.mean(dim=[-1, -2]),
            'R': r_ch.mean(dim=[-1, -2]),
            'S': soil_ch.mean(dim=[-1, -2])
        }
        
        return logits, phys_pred.squeeze(-1), phys_inputs, hidden_states

if __name__ == "__main__":
    print("Running Extended Unit Tests for MFPIT (Forward, Backward, AMP, NaN checks)...")
    from losses import MFPITLoss
    import torch.optim as optim
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from inference import enable_mc_dropout
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else torch.cuda.amp.GradScaler(enabled=False)
    
    model = MFPIT().to(device)
    criterion = MFPITLoss(lambda_phys=0.1, lambda_temp=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Dummy tensor: [B=2, T=12, C=13, H=64, W=64]
    dummy_x = torch.randn(2, 12, 13, 64, 64, device=device)
    dummy_y = torch.randint(0, 2, (2, 1, 64, 64), device=device).float()
    
    # Enable engineering validation mode logic inside the test as requested
    # Zeroing out channel 12 (JRC Occurrence) to prevent leakage
    dummy_x[:, :, CHANNEL_MAP['jrc_occurrence'], :, :] = 0.0
    
    optimizer.zero_grad()
    
    # Forward Pass with AMP
    autocast_context = torch.amp.autocast('cuda') if use_amp else torch.amp.autocast('cpu', enabled=False)
    with autocast_context:
        logits, phys_pred, phys_inputs, hidden_states = model(dummy_x)
        
        assert logits.shape == (2, 1, 64, 64), f"Output shape mismatch! Got {logits.shape}"
        assert not torch.isnan(logits).any(), "NaN found in forward pass output (logits)!"
        
        loss, loss_dict = criterion(logits, dummy_y, phys_pred, phys_inputs, hidden_states)
    
    # Backward pass with GradScaler
    scaler.scale(loss).backward()
    
    # Unscale gradients before checking for NaNs
    scaler.unscale_(optimizer)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"BCE: {loss_dict['bce']:.4f}, Dice: {loss_dict['dice']:.4f}")
    print(f"Physics: {loss_dict['phys']:.4f}, Temporal: {loss_dict['temp']:.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    
    # Check Gradients
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                print(f"Non-finite gradient in {name}")
                has_nan_grad = True
    
    assert not has_nan_grad, "Non-finite values found in backward pass gradients!"
    
    # Test MC Dropout Uncertainty (predict twice, output should differ)
    enable_mc_dropout(model)
    
    with torch.no_grad():
        out1, _, _, _ = model(dummy_x)
        out2, _, _, _ = model(dummy_x)
        diff = torch.abs(out1 - out2).sum().item()
        assert diff > 0, "MC Dropout failed: deterministic output during inference."
    
    print("All Unit Tests Passed! Architecture, AMP, and MC Dropout are stable.")
