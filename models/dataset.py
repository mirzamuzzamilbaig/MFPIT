import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import glob

class MFPITDataset(Dataset):
    """
    Lazy loading PyTorch Dataset for MFPIT temporal sequences.
    Implements a smart file cache to support DataLoader(shuffle=True) efficiently,
    preventing temporal correlation bias while maintaining IO performance.
    """
    def __init__(self, data_dir, split="train", patch_size=64, stride=32, cache_size=10):
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        
        # Discover files
        pattern = f"{split}_*.pt"
        self.files = sorted(list(self.data_dir.glob(pattern)))
        
        if not self.files:
            print(f"Warning: No files found for split '{split}' in {self.data_dir}")
        
        self.h = 279
        self.w = 502
        if self.files:
            dummy_data = torch.load(self.files[0], map_location='cpu')
            self.h, self.w = dummy_data['label'].shape
            
        self.n_rows = max(1, (self.h - self.patch_size) // self.stride + 1)
        self.n_cols = max(1, (self.w - self.patch_size) // self.stride + 1)
        self.patches_per_file = self.n_rows * self.n_cols
        
        self.total_patches = len(self.files) * self.patches_per_file
        
        # Smart Dictionary Cache to support Shuffling (LRU style behavior can be added, but simple dict works for ~10-20 files)
        # Each file is ~80MB, so keeping 10 in RAM is ~800MB.
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return self.total_patches

    def _get_file_data(self, file_idx):
        if file_idx in self.cache:
            return self.cache[file_idx]
            
        # Cache miss
        data = torch.load(self.files[file_idx], map_location='cpu')
        
        # Simple LRU eviction if full
        if len(self.cache) >= self.cache_size:
            # Pop the first key (oldest inserted in Python 3.7+)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[file_idx] = data
        return data

    def __getitem__(self, idx):
        file_idx = idx // self.patches_per_file
        patch_idx = idx % self.patches_per_file
        
        data = self._get_file_data(file_idx)
        tensor = data['tensor'] # [12, 13, H, W]
        label = data['label']   # [H, W]
            
        # Calculate spatial coordinates
        r_idx = patch_idx // self.n_cols
        c_idx = patch_idx % self.n_cols
        
        r_start = r_idx * self.stride
        c_start = c_idx * self.stride
        r_end = r_start + self.patch_size
        c_end = c_start + self.patch_size
        
        # Extract patch
        # Explicitly slicing the identical spatial coordinates across all 12 temporal dimensions
        x_patch = tensor[:, :, r_start:r_end, c_start:c_end]
        y_patch = label[r_start:r_end, c_start:c_end]
        
        # Explicit temporal patch alignment assertion
        # Since slicing is [:, :, row_slice, col_slice], PyTorch inherently guarantees alignment.
        assert x_patch.shape[2:] == (self.patch_size, self.patch_size), "Spatial patch dimension mismatch."
        assert x_patch.shape[0] == 12, "Temporal sequence incomplete."
        
        # Return [12, 13, 64, 64] and [1, 64, 64]
        return x_patch, y_patch.unsqueeze(0)
