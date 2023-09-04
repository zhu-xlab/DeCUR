# multimodal pretrain with DeCUR
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
parser.add_argument('--lr', default=0.2, type=float)
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
        model = DeCUR(args).cuda()
    elif args.method == 'BarlowTwins':
        model = BarlowTwins(args).cuda()
    elif args.method == 'VICReg':
        model = VICReg(args).cuda()
    elif args.method == 'CLIP':
        model = CLIP(args).cuda()    
    elif args.method == 'SimCLR':
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
        
        from datasets.SSL4EO.ssl4eo_dataset_lmdb_mm import LMDBDataset
        
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
            dtype1='float32',
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
            cvtransforms.RandomResizedCrop((224,224), scale=(0.5, 1.)),
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
            cvtransforms.RandomResizedCrop((224,224), scale=(0.5, 1.)),
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
                                 #loss1=loss1.item(),
                                 #loss2=loss2.item(),
                                 #loss12=loss12.item(),
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


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


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




''' self-supervised methods '''

''' DeCUR '''
class DeCUR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet50':
            self.backbone_1 = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet50(zero_init_residual=True)
        elif args.backbone == 'resnet18':
            self.backbone_1 = torchvision.models.resnet18(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet18(zero_init_residual=True)
        elif args.backbone == 'mit_b2':
            from models.segformer.encoders.segformer import mit_b2
            self.backbone_1 = mit_b2(num_classes=2048)
            self.backbone_2 = mit_b2(num_classes=2048)
            self.backbone_1.init_weights(pretrained=args.pretrained)
            self.backbone_2.init_weights(pretrained=args.pretrained)
        elif args.backbone == 'mit_b5':
            from models.segformer.encoders.segformer import mit_b5
            self.backbone_1 = mit_b5(num_classes=2048)
            self.backbone_2 = mit_b5(num_classes=2048)
            self.backbone_1.init_weights(pretrained=args.pretrained)
            self.backbone_2.init_weights(pretrained=args.pretrained)
                        
        if args.mode==['s1','s2c']:
            self.backbone_1.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
            self.backbone_2.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
            self.backbone_1.fc = nn.Identity()
            self.backbone_2.fc = nn.Identity()

        # projector
        if args.backbone == 'resnet50':
            sizes = [2048] + list(map(int, args.projector.split('-')))
        elif args.backbone == 'resnet18':
            sizes = [512] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector1 = nn.Sequential(*layers)
        self.projector2 = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def bt_loss_cross(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size*4)
        torch.distributed.all_reduce(c)

        dim_c = self.args.dim_common
        c_c = c[:dim_c,:dim_c]
        c_u = c[dim_c:,dim_c:]

        on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()
        off_diag_c = off_diagonal(c_c).pow_(2).sum()
        
        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()
        
        loss_c = on_diag_c + self.args.lambd * off_diag_c
        loss_u = on_diag_u + self.args.lambd * off_diag_u
        
        return loss_c,on_diag_c,off_diag_c,loss_u,on_diag_u,off_diag_u   


    def bt_loss_single(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size*4)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss,on_diag,off_diag


    def forward(self, y1_1,y1_2,y2_1,y2_2):
        z1_1 = self.projector1(self.backbone_1(y1_1))
        z1_2 = self.projector1(self.backbone_1(y1_2))
        z2_1 = self.projector2(self.backbone_2(y2_1))
        z2_2 = self.projector2(self.backbone_2(y2_2))        

        loss1, on_diag1, off_diag1 = self.bt_loss_single(z1_1,z1_2)
        loss2, on_diag2, off_diag2 = self.bt_loss_single(z2_1,z2_2)        
        loss12_c, on_diag12_c, off_diag12_c, loss12_u, on_diag12_u, off_diag12_u = self.bt_loss_cross(z1_1,z2_1)
        loss12 = (loss12_c + loss12_u) / 2.0

        return loss1,loss2,loss12,on_diag12_c


''' BarlowTwins '''
class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet50':
            self.backbone_1 = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet50(zero_init_residual=True)
        elif args.backbone == 'resnet18':
            self.backbone_1 = torchvision.models.resnet18(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet18(zero_init_residual=True)
            
        if args.mode==['s1','s2c']:
            self.backbone_1.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
            self.backbone_2.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            self.backbone_1.fc = nn.Identity()
            self.backbone_2.fc = nn.Identity()

        # projector
        if args.backbone == 'resnet50':
            sizes = [2048] + list(map(int, args.projector.split('-')))
        elif args.backbone == 'resnet18':
            sizes = [512] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector1 = nn.Sequential(*layers)
        self.projector2 = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector1(self.backbone_1(y1))
        z2 = self.projector2(self.backbone_2(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size*4)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss#,on_diag,off_diag


''' VICReg '''
class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet50':
            self.backbone_1 = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet50(zero_init_residual=True)
        elif args.backbone == 'resnet18':
            self.backbone_1 = torchvision.models.resnet18(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet18(zero_init_residual=True)
            
        if args.mode==['s1','s2c']:
            self.backbone_1.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
            self.backbone_2.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            self.backbone_1.fc = nn.Identity()
            self.backbone_2.fc = nn.Identity()

        # projector
        if args.backbone == 'resnet50':
            sizes = [2048] + list(map(int, args.projector.split('-')))
        elif args.backbone == 'resnet18':
            sizes = [512] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector1 = nn.Sequential(*layers)
        self.projector2 = nn.Sequential(*layers)
        
        self.num_features = sizes[-1]

    def forward(self, x, y):
        x = self.projector1(self.backbone_1(x))
        y = self.projector2(self.backbone_2(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss        
        
class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]        


''' CLIP '''
class CLIP(nn.Module):

    LARGE_NUMBER = 1e9

    def __init__(self,args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet50':
            self.backbone_1 = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet50(zero_init_residual=True)
        elif args.backbone == 'resnet18':
            self.backbone_1 = torchvision.models.resnet18(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet18(zero_init_residual=True)
            
        if args.mode==['s1','s2c']:
            self.backbone_1.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
            self.backbone_2.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            self.backbone_1.fc = nn.Identity()
            self.backbone_2.fc = nn.Identity()
                    
            self.projector1 = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128), nn.BatchNorm1d(128))
            self.projector2 = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128), nn.BatchNorm1d(128))
            
        self.tau = 1.0
        self.multiplier = 2
        self.distributed = True
        self.temperature = 1.0               
        
    def forward(self,x1,x2):
        z1 = self.projector1(self.backbone_1(x1))
        z2 = self.projector2(self.backbone_2(x2))        
        z = torch.cat((z1,z2),dim=0)

        n = z.shape[0]
        assert n % self.multiplier == 0
        
        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)
        
        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]
        
        z1_new = z[:n//2]
        z2_new = z[n//2:]
        
        logits = (z1_new @ z2_new.T) / self.temperature
        targets = torch.arange(n//2).cuda()
        loss = torch.nn.CrossEntropyLoss()(logits, targets)
                     
        return loss


''' SimCLR '''
class SimCLR(nn.Module):

    LARGE_NUMBER = 1e9

    def __init__(self,args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet50':
            self.backbone_1 = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet50(zero_init_residual=True)
        elif args.backbone == 'resnet18':
            self.backbone_1 = torchvision.models.resnet18(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet18(zero_init_residual=True)
            
        if args.mode==['s1','s2c']:
            self.backbone_1.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
            self.backbone_2.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            self.backbone_1.fc = nn.Identity()
            self.backbone_2.fc = nn.Identity()
                    
            self.projector1 = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128), nn.BatchNorm1d(128))
            self.projector2 = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128), nn.BatchNorm1d(128))
            
        self.tau = 1.0
        self.multiplier = 2
        self.distributed = True
        self.norm = 1.0            
        
    def forward(self,x1,x2):
        z1 = self.projector1(self.backbone_1(x1))
        z2 = self.projector2(self.backbone_2(x2))        
        z = torch.cat((z1,z2),dim=0)

        n = z.shape[0]
        assert n % self.multiplier == 0
        
        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)
        
        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]
        
        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -1e4

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -1e4
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)                
    
        return loss#, acc.sum()/(acc.shape[0])


if __name__ == '__main__':
    main()