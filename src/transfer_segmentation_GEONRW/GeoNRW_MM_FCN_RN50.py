import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models

## change01 ##
from cvtorchvision import cvtransforms
import time
import os
import math
import pdb
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import numpy as np
import argparse
import builtins

from datasets.GeoNRW.geonrw_dataset import GeoNRWDataset, random_subset

from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import create_feature_extractor
#import kornia.augmentation as K
import torchmetrics

parser = argparse.ArgumentParser()
### todo
parser.add_argument('--data1', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data2', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--mask', type=str, metavar='DIR',
                    help='path to dataset')                    
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/resnet/')
parser.add_argument('--resume', type=str, default='')
#parser.add_argument('--save_path', type=str, default='./checkpoints/bigearthnet_s2_B12_100_no_pretrain_resnet50.pt')

#parser.add_argument('--bands', type=str, default='all', choices=['all','RGB'], help='bands to process')  
parser.add_argument('--train_frac', type=float, default=1.0)
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pretrained', default='', type=str, help='path to moco pretrained checkpoint')

### distributed running ###
parser.add_argument('--dist_url', default='env://', type=str)
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

parser.add_argument('--normalize',action='store_true',default=False)
parser.add_argument('--linear',action='store_true',default=False)

parser.add_argument('--mode', nargs='*', default=['RGB','DSM', 'mask'], help='bands to process')
parser.add_argument('--test',action='store_true',default=False)

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

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():

    global args
    args = parser.parse_args()
    ### dist ###
    init_distributed_mode(args)
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    fix_random_seeds(args.seed)

    if args.rank==0 and not os.path.isdir(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoints_dir,'log'))

    '''
    ## change03 ##
    transforms_geometry = K.AugmentationSequential(
        K.RandomResizedCrop((224, 224), scale=(0.5, 1.0), p=1.0),
        K.RandomHorizontalFlip(p=0.5),
        #K.RandomVerticalFlip(p=0.5),
        #K.RandomRotation(degrees=45,resample=Resample.NEAREST, p=0.5),
        #K.RandomAffine((-15., 20.), resample=Resample.NEAREST, p=0.5),
    )
    
    transforms_color = K.AugmentationSequential(
        #K.ColorJitter(0.4,0.4,0.4,0.4,p=0.5),
        #K.RandomGrayscale(p=0.5),
        K.RandomGaussianBlur((3,3),(0.1,2.0),p=0.01),
    )
    '''



    train_dataset = GeoNRWDataset(
        rgb_dir=os.path.join(args.data1,'train'),
        dsm_dir=os.path.join(args.data2,'train'),
        mask_dir = os.path.join(args.mask,'train'),
        #rgb_transform=transforms_color,
        #transform=transforms_geometry,
        mode=args.mode
    )
    ### todo
    val_dataset = GeoNRWDataset(
        rgb_dir=os.path.join(args.data1,'test'),
        dsm_dir=os.path.join(args.data2,'test'),
        mask_dir = os.path.join(args.mask,'test'),
        #rgb_transform=transforms_color,
        #transform=transforms_geometry,
        mode=args.mode
    )    

        
    if args.train_frac is not None and args.train_frac<1:
        train_dataset = random_subset(train_dataset,args.train_frac,args.seed)    
    ### dist ###    
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)    
        
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batchsize,
                              sampler = sampler,
                              #shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=args.is_slurm_job, # improve a little when using lmdb dataset
                              drop_last=True                              
                              )
                              
    val_loader = DataLoader(val_dataset,
                              batch_size=args.batchsize,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=args.is_slurm_job, # improve a little when using lmdb dataset
                              drop_last=True                              
                              )
    
    print('train_len: %d val_len: %d' % (len(train_dataset),len(val_dataset)))

    ## change 04 ##
    if args.backbone == 'resnet50':
        net = FCN_RN50()       
            
    if args.linear:
        for name, param in net.named_parameters():
            if 'backbone_1' in name or 'backbone_2' in name:
                param.requires_grad = False           

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['model']
            state_dict = {k.replace("module.backbone_1", "backbone_1"): v for k,v in state_dict.items()}
            state_dict = {k.replace("module.backbone_2", "backbone_2"): v for k,v in state_dict.items()}
            msg = net.load_state_dict(state_dict, strict=False)                                   
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # convert batch norm layers (if any)
    if args.is_slurm_job:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net.cuda()
        net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.gpu_to_work_on],find_unused_parameters=True)                    
    else:
        net.cuda()                
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    # optimizer = torch.optim.adamw(net.parameters(),lr=args.lr)


    last_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['model_state_dict']
        #state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
        net.load_state_dict(state_dict)            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        last_loss = checkpoint['loss']


    if args.test:
        print('Start testing...')
        running_loss_val = 0.0
        running_acc_val = 0.0
        running_miou_val = 0.0
        count_val = 0
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(val_loader, 0):

                inputs1, inputs2, labels = data[0].cuda(), data[1].cuda(), data[2].cuda()
                
                outputs = net(inputs1,inputs2)
                    
                loss_val = criterion(outputs, labels) 
                val_miou = torchmetrics.functional.jaccard_index(torch.argmax(outputs,1).detach(), labels.detach(), task="multiclass", num_classes=11, ignore_index=0)
                val_acc = torchmetrics.functional.accuracy(torch.argmax(outputs,1).detach(), labels.detach(), task="multiclass", num_classes=11, ignore_index=0)
                count_val += 1
                running_loss_val += loss_val.item()
                running_miou_val += val_miou
                running_acc_val += val_acc       

        print('val_loss: %.3f val_acc: %.3f val_miou: %.3f time: %s seconds.' % (running_loss_val/count_val, running_acc_val/count_val, running_miou_val/count_val, time.time()-start_time))    
    

    #pdb.set_trace()

    print('Start training...')
    for epoch in range(last_epoch,args.epochs):

        net.train()
        adjust_learning_rate(optimizer, epoch, args)
        
        train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        running_acc = 0.0
        running_miou = 0.0
        
        running_loss_epoch = 0.0
        running_acc_epoch = 0.0
        running_miou_epoch = 0.0
        
        start_time = time.time()
        end = time.time()
        sum_bt = 0.0
        sum_dt = 0.0
        sum_tt = 0.0
        sum_st = 0.0
        for i, data in enumerate(train_loader, 0):
            data_time = time.time()-end

            inputs1, inputs2, labels = data[0].cuda(), data[1].cuda(), data[2].cuda()
                       
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs1,inputs2)
            #pdb.set_trace()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_time = time.time()-end-data_time
            
            
            # print statistics
            train_miou = torchmetrics.functional.jaccard_index(torch.argmax(outputs,1).detach(), labels.detach(), task="multiclass", num_classes=11, ignore_index=0)
            train_acc = torchmetrics.functional.accuracy(torch.argmax(outputs,1).detach(), labels.detach(), task="multiclass", num_classes=11, ignore_index=0)
            
            
            running_loss += loss.item()
            running_acc += train_acc
            running_miou += train_miou
            
            score_time = time.time()-end-data_time-train_time
            batch_time = time.time() - end
            end = time.time()        
            sum_bt += batch_time
            sum_dt += data_time
            sum_tt += train_time
            sum_st += score_time
            
            if i % 20 == 19:    # print every 20 mini-batches

                print('[%d, %5d] loss: %.3f acc: %.3f miou: %.3f batch_time: %.3f data_time: %.3f train_time: %.3f score_time: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20, running_acc / 20, running_miou/20, sum_bt/20, sum_dt/20, sum_tt/20, sum_st/20))
                
                #train_iter =  i*args.batch_size / len(train_dataset)
                #tb_writer.add_scalar('train_loss', running_loss/20, global_step=(epoch+1+train_iter) )
                running_loss_epoch = running_loss/20
                running_acc_epoch = running_acc/20
                running_miou_epoch = running_miou/20
                
                
                running_loss = 0.0
                running_acc = 0.0
                running_miou = 0.0
                sum_bt = 0.0
                sum_dt = 0.0
                sum_tt = 0.0
                sum_st = 0.0

        if epoch % 5 == 4:
            running_loss_val = 0.0
            running_acc_val = 0.0
            running_miou_val = 0.0
            count_val = 0
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(val_loader, 0):

                    inputs1, inputs2, labels = data[0].cuda(), data[1].cuda(), data[2].cuda()
                    
                    outputs = net(inputs1,inputs2)
                        
                    loss_val = criterion(outputs, labels) 
                    val_miou = torchmetrics.functional.jaccard_index(torch.argmax(outputs,1).detach(), labels.detach(), task="multiclass", num_classes=11, ignore_index=0)
                    val_acc = torchmetrics.functional.accuracy(torch.argmax(outputs,1).detach(), labels.detach(), task="multiclass", num_classes=11, ignore_index=0)
                    count_val += 1
                    running_loss_val += loss_val.item()
                    running_miou_val += val_miou
                    running_acc_val += val_acc       

            print('Epoch %d val_loss: %.3f val_acc: %.3f val_miou: %.3f time: %s seconds.' % (epoch+1, running_loss_val/count_val, running_acc_val/count_val, running_miou_val/count_val, time.time()-start_time))

            if args.rank == 0:
                losses = {'train': running_loss_epoch,
                          'val': running_loss_val/count_val}
                accs = {'train': running_acc_epoch,
                        'val': running_acc_val/count_val}
                mious = {'train': running_miou_epoch,
                        'val': running_miou_val/count_val}                                
                tb_writer.add_scalars('loss', losses, global_step=epoch+1, walltime=None)
                tb_writer.add_scalars('acc', accs, global_step=epoch+1, walltime=None)
                tb_writer.add_scalars('miou', mious, global_step=epoch+1, walltime=None)
        
            
            
        if args.rank==0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':loss,
                        }, os.path.join(args.checkpoints_dir,'checkpoint_{:04d}.pth.tar'.format(epoch)))
        
    #if args.rank==0:
    #    torch.save(net.state_dict(), save_path)
        
    print('Training finished.')



class FCN_RN50(torch.nn.Module):
    def __init__(self):
        super(FCN_RN50, self).__init__()
        # Get a resnet50 backbone
        m1 = models.resnet50()
        m2 = models.resnet50()
        # Extract 4 main layers
        self.backbone_1 = create_feature_extractor(
            m1, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([2, 3, 4])})
        self.backbone_2 = create_feature_extractor(
            m2, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([2, 3, 4])})
        self.uc0 = torch.nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1)
        self.uc1 = torch.nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.uc2 = torch.nn.Conv2d(in_channels=4096, out_channels=512, kernel_size=1)

        self.up0 = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up1 = torch.nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up2 = torch.nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.cls0 = torch.nn.Conv2d(in_channels=128, out_channels=11, kernel_size=1)
        self.cls1 = torch.nn.Conv2d(in_channels=256, out_channels=11, kernel_size=1)
        self.cls2 = torch.nn.Conv2d(in_channels=512, out_channels=11, kernel_size=1)

        #self.softmax = torch.nn.Softmax()

    def forward(self, x1,x2):
        x1 = self.backbone_1(x1)
        x2 = self.backbone_2(x2)
        #pdb.set_trace()
        x12 = {}
        for key in x1.keys():
            x12[key] = torch.cat((x1[key],x2[key]),dim=1)
        y0 = self.cls0(self.up0(self.uc0(x12['0'])))
        y1 = self.cls1(self.up1(self.uc1(x12['1'])))
        y2 = self.cls2(self.up2(self.uc2(x12['2'])))

        y = y0 + y1 + y2

        #y = self.softmax(y)


        #x = self.fpn(x)
        return y


if __name__ == "__main__":
    main()
