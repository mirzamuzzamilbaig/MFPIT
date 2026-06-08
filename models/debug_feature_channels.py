import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import MFPITDataset
from run_ablations import AblationDataset

CHANNELS = [
    (0,  "NDWI"),
    (1,  "MNDWI"),
    (2,  "CHIRPS_precip"),
    (3,  "MODIS_ET"),
    (4,  "TC_tmmx"),
    (5,  "TC_tmmn"),
    (6,  "TC_soil"),
    (7,  "TC_runoff"),
    (8,  "DEM_elevation"),
    (9,  "DEM_slope"),
    (10, "SAR_VV"),
    (11, "SAR_VH"),
    (12, "JRC_occurrence")
]

def verify_ablation_masking(mode):
    print(f"\n--- AUDITING ABLATION MODE: {mode.upper()} ---")
    
    # Load dummy datasets
    base_dataset = MFPITDataset(data_dir="../data/processed/tensors", split="val")
    if len(base_dataset) == 0:
        print("Error: No data available for audit.")
        return
        
    ablation_dataset = AblationDataset(base_dataset, mode)
    x, _ = ablation_dataset[0] # [12, 13, 64, 64]
    
    kept_channels = []
    masked_channels = []
    
    for idx, name in CHANNELS:
        # Check if the entire temporal sequence is zeroed out for this channel index
        is_zero = torch.all(x[:, idx, :, :] == 0.0).item()
        if is_zero:
            masked_channels.append(f"Channel {idx}: {name}")
        else:
            kept_channels.append(f"Channel {idx}: {name}")
            
    print(f"KEPT CHANNELS ({len(kept_channels)}):")
    for ch in kept_channels:
        print(f"  [KEEP] {ch}")
        
    print(f"MASKED CHANNELS ({len(masked_channels)}):")
    for ch in masked_channels:
        print(f"  [MASK] {ch}")

def run_channel_audit():
    print("=================================================================")
    print("--- STEP 0: RIGOROUS SCIENTIFIC CHANNEL AUDIT ---")
    print("=================================================================")
    
    modes = ["static_full", "terrain_only", "no_jrc", "dynamic_only"]
    for mode in modes:
        verify_ablation_masking(mode)
        
    print("\n=================================================================")
    print("Audit Complete. Ready to execute with absolute indexing confidence.")
    print("=================================================================")

if __name__ == "__main__":
    run_channel_audit()
