# adjusted from SeCo

import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lmdb
from tqdm import tqdm

# VH, VV
MEAN_S1 = [-19.22, -12.59]
STD_S1 = [5.42, 5.04]

def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

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


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def make_lmdb(dataset, lmdb_file, num_workers=6):
    loader = InfiniteDataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    env = lmdb.open(lmdb_file, map_size=1099511627776)

    txn = env.begin(write=True)
    for index, (sample, target) in tqdm(enumerate(loader), total=len(dataset), desc='Creating LMDB'):
        sample = np.array(sample)
        obj = (sample.tobytes(), sample.shape, target.tobytes())
        txn.put(str(index).encode(), pickle.dumps(obj))
        if index % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    env.sync()
    env.close()


class LMDBDataset(Dataset):

    def __init__(self, lmdb_file, is_slurm_job=False, transform=None, s1_transform=None, s2_transform=None, target_transform=None):
        self.lmdb_file = lmdb_file
        self.transform = transform
        self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        self.target_transform = target_transform
        self.is_slurm_job = is_slurm_job

        if not self.is_slurm_job:
            self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.length = txn.stat()['entries']            
        else:
            # Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time
            self.env = None
            if 'train' in self.lmdb_file:
                self.length = 300000
            elif 'val' in self.lmdb_file:
                self.length = 100000
            else:
                raise NotImplementedError

    def _init_db(self):
        
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index):
        if self.is_slurm_job:
            # Delay loading LMDB data until after initialization
            if self.env is None:
                self._init_db()
        
        with self.env.begin(write=False) as txn:
            data = txn.get(str(index).encode())

        sample_s2_bytes, sample_s2_shape, sample_s1_bytes, sample_s1_shape, target_bytes = pickle.loads(data)
        sample_s2 = np.frombuffer(sample_s2_bytes, dtype=np.uint8).reshape(sample_s2_shape)
        sample_s1 = np.frombuffer(sample_s1_bytes, dtype=np.float32).reshape(sample_s1_shape)
            
        
        ### normalize s1
        self.max_q = np.quantile(sample_s1.reshape(-1,2),0.99,axis=0) # VH,VV       
        self.min_q = np.quantile(sample_s1.reshape(-1,2),0.01,axis=0) # VH,VV
        img_bands = []
        for b in range(2):
            img = sample_s1[:,:,b].copy()
            ## outlier
            max_q = self.max_q[b]
            min_q = self.min_q[b]            
            img[img>max_q] = max_q
            img[img<min_q] = min_q
            ## normalize
            img = normalize(img,MEAN_S1[b],STD_S1[b])         
            img = img.reshape(128,128,1)
            img_bands.append(img)
        sample_s1 = np.concatenate((img_bands[1],img_bands[0]),axis=2) # VH,VV
        
        sample = np.concatenate([sample_s2, sample_s1], axis=2) # VV,VH

        target = np.frombuffer(target_bytes, dtype=np.float32)        
        '''
        if self.transform is not None:
            sample_s2 = self.transform(sample_s2)
            sample_s1 = self.transform(sample_s1)
        if self.s1_transform is not None:
            sample_s1 = self.s1_transform(sample_s1)        
        if self.s2_transform is not None:
            sample_s2 = self.s2_transform(sample_s2)        
        if self.target_transform is not None:
            target = self.target_transform(target)
        '''
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import os
    import argparse
    import time
    import torch
    from torchvision import transforms
    from cvtorchvision import cvtransforms
    import cv2
    import random


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--train_frac', type=float, default=1.0)
    args = parser.parse_args()

    test_loading_time = False
    seed = 42


    ### check data augmentation
    
    class GaussianBlur(object):
    #Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    

        def __init__(self, sigma=[.1, 2.]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            
            return cv2.GaussianBlur(x,(3,3),sigma)

    augmentation = [
        #cvtransforms.Resize((128, 128)),
        cvtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        cvtransforms.RandomApply([
            cvtransforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        cvtransforms.RandomGrayscale(p=0.2),
        cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.ToTensor(),
        #normalize
    ]
    train_transforms = cvtransforms.Compose(augmentation)
    
    train_dataset = LMDBDataset(
        lmdb_file=os.path.join(args.data_dir, 'train.lmdb'),
        transform=train_transforms
    )


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=4)    
    for idx, (img,target) in enumerate(train_loader):
        print(idx)


    '''
    ### check loading time
    if test_loading_time:
        train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                               transforms.ToTensor()])
        train_dataset = LMDBDataset(
            lmdb_file=os.path.join(args.data_dir, 'train.lmdb'),
            transform=train_transforms
        )
        
        
        if args.train_frac is not None and args.train_frac < 1:
            train_dataset = random_subset(train_dataset, args.train_frac, seed)        
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=4)    
        start_time = time.time()

        runs = 5
        for i in range(runs):
            for idx, (img,target) in enumerate(train_loader):
                print(idx)
                if idx > 188:
                    break

        print("Mean Time over 5 runs: ", (time.time() - start_time) / runs)
        '''