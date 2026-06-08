import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(-1)
        union = probs_flat.sum(-1) + targets_flat.sum(-1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class MFPITLoss(nn.Module):
    """
    Combined loss for MFPIT model:
    L_total = BCEWithLogits(pos_weight) + DiceLoss + lambda1 * PhysicsLoss + lambda2 * TemporalLoss
    """
    def __init__(self, lambda_phys=0.1, lambda_temp=0.1, pos_weight=None, balance_mu=0.0, balance_sigma=1.0):
        super().__init__()
        # Use BCE with pos_weight for heavy imbalance handling
        # register_buffer ensures automatic device transfer with model.to(device)
        if pos_weight is not None:
            self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight]))
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
        else:
            self.bce = nn.BCEWithLogitsLoss()
            
        self.dice = DiceLoss()
        self.lambda_phys = lambda_phys
        self.lambda_temp = lambda_temp
        
        # Training-set consistent normalization stats for the mass balance
        self.balance_mu = balance_mu
        self.balance_sigma = balance_sigma

    def forward(self, logits, targets, phys_pred, phys_inputs, hidden_states):
        """
        logits: [B, 1, H, W] final predictions
        targets: [B, 1, H, W] ground truth (proxy)
        phys_pred: [B] predicted latent storage proxy from physics branch
        phys_inputs: dictionary of physical variables {P, ET, R} each aggregated
        hidden_states: [B, T, F] hidden temporal states for consistency check
        """
        device = logits.device
        
        # 1. Base Segmentation Loss (BCE + Dice)
        l_bce = self.bce(logits, targets)
        l_dice = self.dice(logits, targets)
        l_seg = 0.5 * l_bce + 0.5 * l_dice
        
        # 2. Proxy Hydrological Consistency Loss (Physics Loss)
        # We explicitly normalize the batch balance proxy using consistent training-set stats
        P = phys_inputs['P'].to(device)
        ET = phys_inputs['ET'].to(device)
        R = phys_inputs['R'].to(device)
        
        raw_balance = (P - ET - R)
        normalized_balance_proxy = (raw_balance - self.balance_mu) / (self.balance_sigma + 1e-8)
        
        # phys_pred predicts the normalized latent storage proxy
        l_phys = torch.mean(torch.abs(phys_pred - normalized_balance_proxy))
        
        # 3. Temporal Consistency Loss with Bounded Inverse Rainfall Weighting
        # ||h_t - h_{t-1}||₂ * exp(-alpha * rain_norm)
        if hidden_states is not None and hidden_states.size(1) > 1:
            diff = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
            # Normalize by square root of features to make it invariant to model width (e.g. 256)
            norm_diff = torch.norm(diff, p=2, dim=-1) / (diff.shape[-1] ** 0.5) # [B, T-1]
            
            # bounded weighting function: w = exp(-alpha * rain_norm)
            # using the P value as the proxy for rain intensity
            # shape of P should align with sequence, but here we use the global P mean as a batch scaler
            # or extract monthly P. Let's use a simple global scalar per sequence for demonstration
            alpha = 0.5
            rain_norm = torch.clamp(P, min=0.0)
            weight = torch.exp(-alpha * rain_norm).unsqueeze(-1) # [B, 1]
            
            l_temp = torch.mean(weight * norm_diff)
        else:
            l_temp = torch.tensor(0.0, device=device)
            
        l_total = l_seg + self.lambda_phys * l_phys + self.lambda_temp * l_temp
        
        return l_total, {
            'bce': l_bce.item(),
            'dice': l_dice.item(),
            'phys': l_phys.item(),
            'temp': l_temp.item(),
            'total': l_total.item()
        }
