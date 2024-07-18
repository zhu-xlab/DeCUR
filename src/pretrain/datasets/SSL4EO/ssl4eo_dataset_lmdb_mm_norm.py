import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lmdb
from tqdm import tqdm
import pdb
#import kornia as K

### band statistics: mean & std
S1_MEAN = [-12.59, -20.26]
S1_STD = [5.26, 5.91]

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


def make_lmdb(dataset, lmdb_file, num_workers=6,mode=['s1','s2c']):
    loader = InfiniteDataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    #env = lmdb.open(lmdb_file, map_size=1099511627776,writemap=True) # continuously write to disk
    env = lmdb.open(lmdb_file, map_size=1099511627776)
    txn = env.begin(write=True)
    for index, (s1, s2c) in tqdm(enumerate(loader), total=len(dataset), desc='Creating LMDB'):
        if 's1' in mode:
            sample_s1 = np.array(s1)
        if 's2c' in mode:
            sample_s2c = np.array(s2c)
            
        if mode==['s1','s2c']:
            obj = (sample_s1.tobytes(), sample_s1.shape, sample_s2c.tobytes(), sample_s2c.shape)
        elif mode==['s1']:
            obj = (sample_s1.tobytes(), sample_s1.shape)
        elif mode==['s2a']:
            obj = (sample_s2a.tobytes(), sample_s2a.shape)
        elif mode==['s2c']:
            obj = (sample_s2c.tobytes(), sample_s2c.shape)
            
        txn.put(str(index).encode(), pickle.dumps(obj))            

        if index % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    env.sync()
    env.close()


class LMDBDataset(Dataset):

    def __init__(self, lmdb_file_s1, lmdb_file_s2, is_slurm_job=False, s1_transform=None, s2c_transform=None, subset=None, normalize=False, mode=['s1','s2c'], dtype1='float32', dtype2='uint8'):
        self.lmdb_file_s1 = lmdb_file_s1
        self.lmdb_file_s2 = lmdb_file_s2
        self.s1_transform = s1_transform
        self.s2c_transform = s2c_transform
        self.is_slurm_job = is_slurm_job
        self.subset = subset
        self.normalize = normalize
        self.mode = mode
        self.dtype1 = dtype1
        self.dtype2 = dtype2

        if not self.is_slurm_job:
            self.env1 = lmdb.open(self.lmdb_file_s1, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with self.env1.begin(write=False) as txn:
                self.length = txn.stat()['entries']
                
            self.env2 = lmdb.open(self.lmdb_file_s2, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
                        
        else:
            # Workaround to have length from the start since we don't have LMDB at initialization time
            self.env1 = None
            self.env2 = None
            
            if self.subset is not None:
                self.length = 50000
            else:
                self.length = 251079

                
                
    def _init_db(self):
        
        self.env1 = lmdb.open(self.lmdb_file_s1, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env1.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        self.env2 = lmdb.open(self.lmdb_file_s2, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.is_slurm_job:
            # Delay loading LMDB data until after initialization
            if self.env1 is None:
                self._init_db()
            if self.env2 is None:
                self._init_db()

        with self.env1.begin(write=False) as txn1:
            data1 = txn1.get(str(index).encode())
        with self.env2.begin(write=False) as txn2:
            data2 = txn2.get(str(index).encode())
        
        ## s1
        if 's1' in self.mode:
            s1_bytes, s1_shape, _, _, _, _ = pickle.loads(data1)
            if self.dtype1=='uint8':
                sample_s1 = np.frombuffer(s1_bytes, dtype=np.uint8).reshape(s1_shape)
            else:
                sample_s1 = np.frombuffer(s1_bytes, dtype=np.float32).reshape(s1_shape)
                #print(sample_s1.shape)
                ### normalize s1
                self.max_q = np.quantile(sample_s1.reshape(-1,2),0.99,axis=0) # VH,VV       
                self.min_q = np.quantile(sample_s1.reshape(-1,2),0.01,axis=0) # VH,VV
                img_seasons = []
                for s in range(4):
                    img_bands = []
                    for b in range(2):
                        img = sample_s1[s,b,:,:].copy()
                        ## outlier
                        max_q = self.max_q[b]
                        min_q = self.min_q[b]            
                        img[img>max_q] = max_q
                        img[img<min_q] = min_q
                        ## normalize
                        img = normalize(img,S1_MEAN[b],S1_STD[b])        
                        #img = img.reshape(264,264,1)
                        img_bands.append(img)
                    img_seasons.append(np.stack(img_bands,0))
                sample_s1 = np.stack(img_seasons,axis=0)
                
                
            if self.s1_transform is not None:
                sample_s1 = self.s1_transform(sample_s1)
        
        ## s2c
        if 's2c' in self.mode:
            s2c_bytes, s2c_shape, _,_,_,_ = pickle.loads(data2)
            if self.dtype2=='uint8':
                sample_s2c = np.frombuffer(s2c_bytes, dtype=np.uint8).reshape(s2c_shape)
            else:
                sample_s2c = np.frombuffer(s2c_bytes, dtype=np.int16).reshape(s2c_shape)
                sample_s2c = (sample_s2c / 10000.0).astype(np.float32)
            if self.s2c_transform is not None:
                sample_s2c = self.s2c_transform(sample_s2c)    
                        
        if self.mode==['s1','s2c']:
            return sample_s1, sample_s2c
        elif self.mode==['s1']:
            return sample_s1
        elif self.mode==['s2c']:
            return sample_s2c
    
    
    def __len__(self):
        return self.length
