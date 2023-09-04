import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb
import os
import cv2
import rasterio
from PIL import Image

class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def random_subset(dataset, frac, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(len(dataset)), int(frac * len(dataset)))
    return Subset(dataset, indices)

class GeoNRWDataset(Dataset):

    def __init__(self, rgb_dir, dsm_dir, mask_dir, mode=['RGB','DSM'], rgb_transform=None, dsm_transform=None, mask_transform=None):
        self.rgb_dir = rgb_dir
        self.dsm_dir = dsm_dir
        self.mask_dir = mask_dir
        self.mode = mode
        
        self.rgb_files = os.listdir(rgb_dir)

        self.length = len(self.rgb_files)
        
        self.rgb_transform = rgb_transform
        self.dsm_transform = dsm_transform
        self.mask_transform = mask_transform
        
    def __getitem__(self, index):
        
        rgb_file = self.rgb_files[index]
        rgb_path = os.path.join(self.rgb_dir,rgb_file)
        dsm_file = rgb_file.replace('rgb.png','dem.tif')
        dsm_path = os.path.join(self.dsm_dir,dsm_file)

        
        #rgb = np.array(Image.open(rgb_file).convert("RGB"))
        rgb = cv2.imread(rgb_path)
        if self.rgb_transform is not None:
            rgb = self.rgb_transform(rgb)
        
        dsm = np.expand_dims(np.array(Image.open(dsm_path)),-1)
        p1,p99 = np.percentile(dsm,(1,99))
        dsm = np.clip(dsm,p1,p99)
        dsm = (dsm - p1) / (p99-p1+1e-5)
        
        if self.dsm_transform is not None:
            dsm = self.dsm_transform(dsm)  
                  
        if 'mask' in self.mode:
            mask_file = rgb_file.replace('rgb.png','seg.tif')
            mask_path = os.path.join(self.mask_dir,mask_file)
            mask = np.array(Image.open(mask_path))
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            return rgb, dsm, mask
        else:
            mask = None
            return rgb, dsm
        
    def __len__(self):
        return self.length
