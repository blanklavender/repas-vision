# Save this as custom_reader.py
import numpy as np
import cv2
import os
from pathlib import Path

class CustomDataReader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / 'rgb'
        self.depth_dir = self.data_dir / 'depth'
        self.mask_dir = self.data_dir / 'mask'
        
        # Load camera intrinsics
        self.K = np.loadtxt(self.data_dir / 'cam_K.txt')
        
        # Get sorted list of files
        self.color_files = sorted(self.rgb_dir.glob('*.png'))
        self.depth_files = sorted(self.depth_dir.glob('*.png'))
        
        # Create ID strings for saving
        self.id_strs = [f.stem for f in self.color_files]
        
        print(f"Loaded {len(self.color_files)} frames")
    
    def __len__(self):
        return len(self.color_files)
    
    def get_color(self, idx):
        """Load RGB image"""
        img = cv2.imread(str(self.color_files[idx]))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def get_depth(self, idx):
        """Load depth image
        Assumes depth is in millimeters (uint16)
        Convert to meters (float)
        """
        depth = cv2.imread(str(self.depth_files[idx]), cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 1000.0  # mm to meters
    
    def get_mask(self, idx):
        """Load mask for first frame"""
        mask_file = self.mask_dir / f'{self.id_strs[idx]}.png'
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            return (mask > 0).astype(np.uint8)
        else:
            # If no mask, return full image
            color = self.get_color(idx)
            return np.ones(color.shape[:2], dtype=np.uint8)