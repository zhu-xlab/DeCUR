import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb
import os
import cv2
import rasterio
from PIL import Image
import kornia as K
from einops import rearrange

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


class MyAugmentation(torch.nn.Module):
  def __init__(self):
    super(MyAugmentation, self).__init__()
    # we define and cache our operators as class members
    #self.k1 = K.augmentation.ColorJitter(0.15, 0.25, 0.25, 0.25)
    self.k1 = K.augmentation.RandomResizedCrop((224,224), scale=(0.2, 1.0), resample = 'nearest', align_corners=None)
    self.k2 = K.augmentation.RandomHorizontalFlip(p=0.5)
  
  def forward(self, rgb: torch.Tensor, dsm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # 1. apply color only in image
    # 2. apply geometric tranform
    rgb = self.k1(rgb)
    dsm = self.k1(dsm, self.k1._params)
    mask = self.k1(mask, self.k1._params)
    
    rgb = self.k2(rgb)
    dsm = self.k2(dsm, self.k2._params)
    mask = self.k2(mask, self.k2._params)

    return rgb, dsm, mask # C,H,W


class GeoNRWDataset(Dataset):

    def __init__(self, rgb_dir, dsm_dir, mask_dir, mode=['RGB','DSM','mask']):
        self.rgb_dir = rgb_dir
        self.dsm_dir = dsm_dir
        self.mask_dir = mask_dir
        self.mode = mode
        
        self.rgb_files = os.listdir(rgb_dir)
        
        self.length = len(self.rgb_files)
        
        self.transform = MyAugmentation()
        
    def __getitem__(self, index):
        
        rgb_file = self.rgb_files[index]
        rgb_path = os.path.join(self.rgb_dir,rgb_file)
        dsm_file = rgb_file.replace('rgb.png','dem.tif')
        dsm_path = os.path.join(self.dsm_dir,dsm_file)
        
        
        #rgb = np.array(Image.open(rgb_file).convert("RGB"))
        rgb = cv2.imread(rgb_path) / 255.0 # 250,250,3
        rgb = torch.from_numpy(rgb.transpose(2,0,1).astype('float32')) # 3,250,250
        
        dsm = np.array(Image.open(dsm_path))    
        dsm = np.stack((dsm,dsm,dsm),0)
        dsm = (dsm - dsm.min()) / (dsm.max()-dsm.min())
        dsm = torch.from_numpy(dsm.astype('float32')) # 3,250,250
                  
        if 'mask' in self.mode:
            mask_file = rgb_file.replace('rgb.png','seg.tif')
            mask_path = os.path.join(self.mask_dir,mask_file)
            mask = np.array(Image.open(mask_path))
            mask = torch.from_numpy(mask.astype('float32')) # 250,250        
        
        if self.transform is not None:
            rgb, dsm, mask = self.transform(rgb, dsm, mask)
            rgb = rearrange(rgb, "() c h w -> c h w")
            dsm = rearrange(dsm, "() c h w -> c h w")
            mask = rearrange(mask, "() () h w -> h w")            
        
        return rgb, dsm, mask.long()


        
    def __len__(self):
        return self.length
