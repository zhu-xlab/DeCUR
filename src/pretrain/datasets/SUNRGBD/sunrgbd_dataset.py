import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb
import os
import cv2
import rasterio
from PIL import Image
import glob

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

class SUNRGBDDataset(Dataset):

    def __init__(self, rgb_dir, depth_dir, mask_dir, mode=['rgb','depth'], rgb_transform=None, depth_transform=None, mask_transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.mode = mode
        
        self.rgb_paths = glob.glob(os.path.join(rgb_dir,'*.jpg'))
        self.length = len(self.rgb_paths)
        
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.mask_transform = mask_transform
        
    def __getitem__(self, index):
        
        rgb_path = self.rgb_paths[index]
        
        depth_path = rgb_path.replace(self.rgb_dir,self.depth_dir)
        if not os.path.isfile(depth_path):
            depth_path = depth_path.replace('.jpg','.png')
        
        #rgb = np.array(Image.open(rgb_file).convert("RGB"))
        rgb = cv2.imread(rgb_path)
        if self.rgb_transform is not None:
            rgb = self.rgb_transform(rgb)
        
        #dsm = np.expand_dims(np.array(Image.open(depth_path)),-1)
        #p1,p99 = np.percentile(dsm,(1,99))
        #dsm = np.clip(dsm,p1,p99)
        #dsm = (dsm - p1) / (p99-p1+1e-5)
        #dsm = dsm.astype('float32')
        dsm = cv2.imread(depth_path)
        
        if self.depth_transform is not None:
            dsm = self.depth_transform(dsm)  
                  
        if 'mask' in self.mode:
            mask_path = rgb_path.replace(self.rgb_dir,self.mask_dir).replace('jpg','png')
            mask = np.array(Image.open(mask_path))
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            return rgb, dsm, mask
        else:
            mask = None
            return rgb, dsm
        
    def __len__(self):
        return self.length