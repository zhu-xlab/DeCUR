# multimodal self-supervised learning with DeCUR
# Adapted from https://github.com/facebookresearch/barlowtwins

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.functional as F
import diffdist


from cvtorchvision import cvtransforms
from utils.rs_transforms_uint8 import RandomChannelDrop,RandomBrightness,RandomContrast,ToGray,GaussianBlur,Solarize
import pdb

parser = argparse.ArgumentParser(description='Multimodal self-supervised pretraining')
parser.add_argument('--dataset', type=str,
                    help='pretraining dataset', choices=['SSL4EO','GEONRW','SUNRGBD'])
parser.add_argument('--method', type=str,
                    help='pretraining method', choices=['DeCUR','CLIP','SimCLR','BarlowTwins','VICReg'])                    
parser.add_argument('--data1', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data2', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--lr', default=0.2, type=float) # no effect
parser.add_argument('--cos', action='store_true', default=False)
parser.add_argument('--schedule', default=[120,160], nargs='*', type=int)

parser.add_argument('--mode', nargs='*', default=['s1','s2c'], help='bands to process')
parser.add_argument('--train_frac', type=float, default=1.0)
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--resume', type=str, default='',help='resume path.')
parser.add_argument('--dim_common', type=int, default=448)

parser.add_argument('--pretrained', type=str, default='',help='pretrained path.')

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--rda', action='store_true', default=False) # only available for ResNets and DeCUR


def init_distributed_mode(args):

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])


    # prepare distributed
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return    

def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def main():
    global args
    args = parser.parse_args()

    init_distributed_mode(args)
    
    fix_random_seeds(args.seed)
    
    main_worker(gpu=None,args=args)



def main_worker(gpu, args):

    # create tb_writer
    if args.rank==0 and not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir,exist_ok=True)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoint_dir,'log'))

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    '''
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    '''
    
    ### choose which method for pretraining
    if args.method == 'DeCUR':
        from models.decur import DeCUR
        model = DeCUR(args).cuda()
    elif args.method == 'BarlowTwins':
        from models.barlowtwins import BarlowTwins
        model = BarlowTwins(args).cuda()
    elif args.method == 'VICReg':
        from models.vicreg import VICReg
        model = VICReg(args).cuda()
    elif args.method == 'CLIP':
        from models.clip import CLIP
        model = CLIP(args).cuda()    
    elif args.method == 'SimCLR':
        from models.simclr import SimCLR
        model = SimCLR(args).cuda()        
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu_to_work_on],find_unused_parameters=True)
    
    if 'vit' or 'mit' in args.backbone or args.rda:
        optimizer = torch.optim.AdamW(parameters, args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=True,
                        lars_adaptation_filter=True)
    
    '''
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    '''

    # automatically resume from checkpoint if it exists
    if args.resume:
        ckpt = torch.load(args.resume,
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0    
    
    ### choose which dataset for pretraining
    if args.dataset == 'SSL4EO':
        
        from datasets.SSL4EO.ssl4eo_dataset_lmdb_mm_norm import LMDBDataset
        
        train_transforms_s1 = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            #cvtransforms.RandomApply([
            #    RandomBrightness(0.4),
            #    RandomContrast(0.4)
            #], p=0.8),
            cvtransforms.RandomApply([ToGray(2)], p=0.2),
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.RandomVerticalFlip(),
            #cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),
            cvtransforms.ToTensor()])
        
        train_transforms_s2c = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            cvtransforms.RandomApply([
                RandomBrightness(0.4),
                RandomContrast(0.4)
            ], p=0.8),
            cvtransforms.RandomApply([ToGray(13)], p=0.2),
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            cvtransforms.RandomApply([Solarize()],p=0.2),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.RandomVerticalFlip(),
            #cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),
            cvtransforms.ToTensor()])    
        
        train_dataset = LMDBDataset(
            lmdb_file_s1=args.data1,
            lmdb_file_s2=args.data2,
            s1_transform= TwoCropsTransform_SSL4EO(train_transforms_s1,season='augment'),
            s2c_transform=TwoCropsTransform_SSL4EO(train_transforms_s2c,season='augment'),
            is_slurm_job=args.is_slurm_job,
            normalize=False,
            dtype1='uint8',
            dtype2='uint8',
            mode=args.mode
        ) 
    elif args.dataset == 'GEONRW':
    
        from datasets.GeoNRW.geonrw_dataset import GeoNRWDataset, random_subset
        
        train_transforms_dsm = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            #cvtransforms.RandomApply([
            #    RandomBrightness(0.4),
            #    RandomContrast(0.4)
            #], p=0.8),
            #cvtransforms.RandomApply([ToGray(1)], p=1.0),
            #cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.RandomVerticalFlip(),
            #cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),
            cvtransforms.ToTensor()])
        
        train_transforms_rgb = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            cvtransforms.RandomApply([
                RandomBrightness(0.4),
                RandomContrast(0.4)
            ], p=0.8),
            cvtransforms.RandomApply([ToGray(3)], p=0.2),
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            cvtransforms.RandomApply([Solarize()],p=0.2),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.RandomVerticalFlip(),
            #cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),
            cvtransforms.ToTensor()])    
        
        train_dataset = GeoNRWDataset(
            rgb_dir=args.data1,
            dsm_dir=args.data2,
            mask_dir = None,
            rgb_transform=TwoCropsTransform(train_transforms_rgb),
            dsm_transform=TwoCropsTransform(train_transforms_dsm),
            mode=args.mode
        )
    elif args.dataset == 'SUNRGBD':
    
        from datasets.SUNRGBD.sunrgbd_dataset import SUNRGBDDataset, random_subset
    
        train_transforms_rgb = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224, scale=(0.5, 1.)),
            cvtransforms.RandomApply([
                RandomBrightness(0.4),
                RandomContrast(0.4)
            ], p=0.8),
            cvtransforms.RandomApply([ToGray(3)], p=0.2),
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            cvtransforms.RandomHorizontalFlip(),
            #cvtransforms.RandomVerticalFlip(),
            #cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),
            cvtransforms.ToTensor(),
            cvtransforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        
        train_transforms_dsm = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224, scale=(0.5, 1.)),
            cvtransforms.RandomHorizontalFlip(),
            #cvtransforms.RandomVerticalFlip(),
            cvtransforms.ToTensor(),
            cvtransforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        
        train_dataset = SUNRGBDDataset(
            rgb_dir=args.data1,
            depth_dir=args.data2,
            mask_dir = None,
            rgb_transform=TwoCropsTransform(train_transforms_rgb),
            depth_transform=TwoCropsTransform(train_transforms_dsm),
            mode=args.mode
        )       

        
                  

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=args.is_slurm_job, sampler=train_sampler, drop_last=True)

    print('Start training...')

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)
        for step, (y1, y2) in enumerate(train_loader, start=epoch * len(train_loader)):
            y1_1 = y1[0].cuda(gpu, non_blocking=True)
            y1_2 = y1[1].cuda(gpu, non_blocking=True)
            y2_1 = y2[0].cuda(gpu, non_blocking=True)
            y2_2 = y2[1].cuda(gpu, non_blocking=True)            
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                if args.method=='DeCUR':
                    loss1,loss2,loss12,on_diag12_c = model.forward(y1_1, y1_2, y2_1, y2_2)
                    loss = (loss1 + loss2 + loss12) / 3
                else:
                    loss = model.forward(y1_1, y2_1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #pdb.set_trace()
            
            '''
            loss = model.forward(y1,y2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 #lr=optimizer.param_groups['lr'],
                                 loss=loss.item(),
                                 loss1=loss1.item(),
                                 loss2=loss2.item(),
                                 loss12=loss12.item(),
                                 #on_diag12_c=on_diag12_c.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if args.rank == 0 and epoch%9==0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint_{:04d}.pth'.format(epoch))

            tb_writer.add_scalars('training log',stats,epoch)
            

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    w = 1
    if args.cos:
        w *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            w *= 0.1 if epoch >= milestone else 1.
    optimizer.param_groups[0]['lr'] = w * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = w * args.learning_rate_biases
     

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

class TwoCropsTransform_SSL4EO:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, season='fixed'):
        self.base_transform = base_transform
        self.season = season

    def __call__(self, x):

        if self.season=='augment':
            season1 = np.random.choice([0,1,2,3])
            season2 = np.random.choice([0,1,2,3])
        elif self.season=='fixed':
            np.random.seed(42)
            season1 = np.random.choice([0,1,2,3])
            season2 = season1
        elif self.season=='random':
            season1 = np.random.choice([0,1,2,3])
            season2 = season1

        x1 = np.transpose(x[season1,:,:,:],(1,2,0))
        x2 = np.transpose(x[season2,:,:,:],(1,2,0))

        q = self.base_transform(x1)
        k = self.base_transform(x2)

        return [q, k]
        #return q

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)

        return [q, k]




if __name__ == '__main__':
    main()