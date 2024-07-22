import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from cvtorchvision import cvtransforms
import time
import os
import math
import pdb
#from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_f1_score
import numpy as np
import argparse
import builtins

#from datasets.BigEarthNet.bigearthnet_dataset_seco import Bigearthnet
#from datasets.BigEarthNet.bigearthnet_dataset_lmdb_s2_uint8 import LMDBDataset,random_subset
from datasets.BigEarthNet.bigearthnet_dataset_lmdb_B14 import LMDBDataset, random_subset

from torch.utils.tensorboard import SummaryWriter
import timm

parser = argparse.ArgumentParser()
#parser.add_argument('--data_dir', type=str, default='/mnt/d/codes/SSL_examples/datasets/BigEarthNet')
parser.add_argument('--lmdb_dir', type=str, default='/mnt/d/codes/SSL_examples/datasets/BigEarthNet/dataload_op1_lmdb')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/resnet/')
parser.add_argument('--resume', type=str, default='')

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
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')

### distributed running ###
parser.add_argument('--dist_url', default='env://', type=str)
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

# mode options
parser.add_argument('--normalize',action='store_true',default=False)
parser.add_argument('--linear',action='store_true',default=False)
parser.add_argument('--mode', nargs='*', default=['s1','s2'], help='bands to process')
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

class LateFusionModel(torch.nn.Module):
    def __init__(self,da=False):
        super().__init__()

        self.net1 = timm.create_model('vit_small_patch16_224',pretrained=False)
        self.net1.patch_embed.proj = torch.nn.Conv2d(2, 384, kernel_size=(16, 16), stride=(16, 16))
        self.net1.head = torch.nn.Identity()
        self.net2 = timm.create_model('vit_small_patch16_224',pretrained=False)
        self.net2.patch_embed.proj = torch.nn.Conv2d(13, 384, kernel_size=(16, 16), stride=(16, 16))
        self.net2.head = torch.nn.Identity()

        self.ffc = torch.nn.Linear(768,19)    

    def forward(self,s1,s2):
        z1 = self.net1(s1)
        z2 = self.net2(s2)
        z12 = torch.cat((z1,z2),-1)
        return self.ffc(z12)


class SingleModel(torch.nn.Module):
    def __init__(self,n_channels=2,n_classes=19):
        super().__init__()

        self.n_channels = n_channels

        self.backbone = timm.create_model('vit_small_patch16_224',pretrained=False)
        self.backbone.patch_embed.proj = torch.nn.Conv2d(n_channels, 384, kernel_size=(16, 16), stride=(16, 16))
        self.backbone.head = torch.nn.Linear(384,n_classes)

    def forward(self,s1):
        return self.backbone(s1)




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
    

    lmdb_dir = args.lmdb_dir
    checkpoints_dir = args.checkpoints_dir
    batch_size = args.batchsize
    num_workers = args.num_workers
    epochs = args.epochs
    train_frac = args.train_frac
    seed = args.seed

    if args.rank==0 and not os.path.isdir(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoints_dir,'log'))

    ## change03 ##
    train_transforms = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224,scale=(0.8,1.0)), # multilabel, avoid cropping out labels
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.RandomVerticalFlip(),
            cvtransforms.ToTensor()])

    val_transforms = cvtransforms.Compose([
            cvtransforms.Resize(256),
            cvtransforms.CenterCrop(224),
            cvtransforms.ToTensor(),
            ])


    train_dataset = LMDBDataset(
        lmdb_file=os.path.join(lmdb_dir, 'train_B12_B2.lmdb'),
        transform=train_transforms
    )
    
    val_dataset = LMDBDataset(
        lmdb_file=os.path.join(lmdb_dir, 'val_B12_B2.lmdb'),
        transform=val_transforms
    )    

        
    if train_frac is not None and train_frac<1:
        train_dataset = random_subset(train_dataset,train_frac,seed)    
    ### dist ###    
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)    
        
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler = sampler,
                              #shuffle=True,
                              num_workers=num_workers,
                              pin_memory=args.is_slurm_job, # improve a little when using lmdb dataset
                              drop_last=True                              
                              )
                              
    val_loader = DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=args.is_slurm_job, # improve a little when using lmdb dataset
                              drop_last=True                              
                              )
    
    print('train_len: %d val_len: %d' % (len(train_dataset),len(val_dataset)))

    ## change 04 ##
    if args.backbone == 'vits16':
        if args.mode==['s1']:
            net = SingleModel(n_channels=2,n_classes=19)
            net.backbone.head.weight.data.normal_(mean=0.0,std=0.01)
            net.backbone.head.bias.data.zero_()   
        if args.mode==['s2']:
            net = SingleModel(n_channels=13,n_classes=19)
            net.backbone.head.weight.data.normal_(mean=0.0,std=0.01)
            net.backbone.head.bias.data.zero_()
        if args.mode==['s1','s2']:
            net = LateFusionModel(args.da)            
            net.ffc.weight.data.normal_(mean=0.0,std=0.01)
            net.ffc.bias.data.zero_() 
    if args.linear:
        if args.mode==['s1'] or args.mode==['s2']:
            for name, param in net.named_parameters():
                if name not in ['backbone.head.weight','backbone.head.bias']:
                    param.requires_grad = False
            net.backbone.head.weight.data.normal_(mean=0.0,std=0.01)
            net.backbone.head.bias.data.zero_()    
        if args.mode==['s1','s2']:
            for name, param in net.named_parameters():
                if name not in ['ffc.weight','ffc.bias']:
                    param.requires_grad = False
           

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            if args.mode==['s1']:
                checkpoint1 = torch.load(args.pretrained, map_location="cpu")
                state_dict1 = checkpoint1['model'] if 'model' in checkpoint1 else checkpoint1
                state_dict1 = {k.replace("module.backbone_1", "backbone"): v for k,v in state_dict1.items()}
                state_dict1 = {k.replace("module.", ""): v for k,v in state_dict1.items()}
                msg1 = net.load_state_dict(state_dict1, strict=False)
                assert set(msg1.missing_keys) == {"backbone.head.weight", "backbone.head.bias"}            
            if args.mode==['s2']:
                checkpoint2 = torch.load(args.pretrained, map_location="cpu")
                state_dict2 = checkpoint2['model'] if 'model' in checkpoint2 else checkpoint2
                state_dict2 = {k.replace("module.backbone_2", "backbone"): v for k,v in state_dict2.items()}
                state_dict2 = {k.replace("module.", ""): v for k,v in state_dict2.items()}
                msg2 = net.load_state_dict(state_dict2, strict=False)                       
                assert set(msg2.missing_keys) == {"backbone.head.weight", "backbone.head.bias"}
            if args.mode==['s1','s2']:
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                state_dict = {k.replace("module.backbone_1", "net1"): v for k,v in state_dict.items()}
                state_dict = {k.replace("module.backbone_2", "net2"): v for k,v in state_dict.items()}
                state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
                msg = net.load_state_dict(state_dict, strict=False)                       
                assert set(msg.missing_keys) == {"ffc.weight", "ffc.bias"}            
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
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    last_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['model_state_dict']
        #state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
        net.load_state_dict(state_dict)            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        last_loss = checkpoint['loss']
        print("=> resumed from '{}'".format(args.resume))


    if args.test:
        print("Start testing...")
        start_time = time.time()
        running_loss_val = 0.0
        running_mAP_micro_val = 0.0
        running_mAP_macro_val = 0.0
        #running_f1_micro_val = 0.0
        #running_f1_macro_val = 0.0
        count_val = 0
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(val_loader, 0):
    
                if args.mode==['s1']:
                    images = data[0][:,12:,:,:] # VV,VH
                    inputs_val, labels_val = images.cuda(), data[1].cuda()
                elif args.mode==['s2']:                
                    # pad to 13 dimension
                    b_zeros = torch.zeros((data[0].shape[0],1,data[0].shape[2],data[0].shape[3]),dtype=torch.float32)
                    images = torch.cat((data[0][:,:10,:,:],b_zeros,data[0][:,10:12,:,:]),dim=1)
                    inputs_val, labels_val = images.cuda(), data[1].cuda()
                elif args.mode==['s1','s2']:    
                    images1 = data[0][:,12:,:,:] # VV,VH
                    # pad to 13 dimension
                    b_zeros = torch.zeros((data[0].shape[0],1,data[0].shape[2],data[0].shape[3]),dtype=torch.float32)
                    images2 = torch.cat((data[0][:,:10,:,:],b_zeros,data[0][:,10:12,:,:]),dim=1)
                    inputs1_val, inputs2_val, labels_val = images1.cuda(), images2.cuda(), data[1].cuda()
    
                if args.mode==['s1','s2']:
                    outputs_val = net(inputs1_val,inputs2_val)
                else:
                    outputs_val = net(inputs_val)
                    
                loss_val = criterion(outputs_val, labels_val.long()) 
                score = torch.sigmoid(outputs_val).detach()
                average_precision_micro = multilabel_average_precision(score, labels_val, num_labels=19, average="micro")
                average_precision_macro = multilabel_average_precision(score, labels_val, num_labels=19, average="macro")
                #f1_micro = multilabel_f1_score(score, labels_val, num_labels=19, average="micro")
                #f1_macro = multilabel_f1_score(score, labels_val, num_labels=19, average="macro")
    
    
                count_val += 1
                running_loss_val += loss_val.item()
                running_mAP_micro_val += average_precision_micro
                running_mAP_macro_val += average_precision_macro
                #running_f1_micro_val += f1_micro    
                #running_f1_macro_val += f1_macro    

        print('val_loss: %.4f mAP_micro: %.4f mAP_macro: %.4f time: %.2f' % (running_loss_val/count_val, running_mAP_micro_val/count_val, running_mAP_macro_val/count_val, time.time()-start_time))        
        
    else:
        print('Start training...')
        st_time = time.time()
        for epoch in range(last_epoch,epochs):
    
            net.train()
            adjust_learning_rate(optimizer, epoch, args)
            
            train_loader.sampler.set_epoch(epoch)
            running_loss = 0.0
            running_acc = 0.0
            
            running_loss_epoch = 0.0
            running_acc_epoch = 0.0
            
            start_time = time.time()
            end = time.time()
            sum_bt = 0.0
            sum_dt = 0.0
            sum_tt = 0.0
            sum_st = 0.0
            for i, data in enumerate(train_loader, 0):
                data_time = time.time()-end
                if args.mode==['s1']:
                    images = data[0][:,12:,:,:] # VV,VH
                    inputs, labels = images.cuda(), data[1].cuda()
                elif args.mode==['s2']:                
                    # pad to 13 dimension
                    b_zeros = torch.zeros((data[0].shape[0],1,data[0].shape[2],data[0].shape[3]),dtype=torch.float32)
                    images = torch.cat((data[0][:,:10,:,:],b_zeros,data[0][:,10:12,:,:]),dim=1)
                    inputs, labels = images.cuda(), data[1].cuda()
                elif args.mode==['s1','s2']:    
                    images1 = data[0][:,12:,:,:] # VV,VH
                    # pad to 13 dimension
                    b_zeros = torch.zeros((data[0].shape[0],1,data[0].shape[2],data[0].shape[3]),dtype=torch.float32)
                    images2 = torch.cat((data[0][:,:10,:,:],b_zeros,data[0][:,10:12,:,:]),dim=1)
                    inputs1, inputs2, labels = images1.cuda(), images2.cuda(), data[1].cuda()
                           
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                if args.mode==['s1','s2']:
                    outputs = net(inputs1,inputs2)
                else:
                    outputs = net(inputs)
                #pdb.set_trace()
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                train_time = time.time()-end-data_time
    
                score = torch.sigmoid(outputs).detach()
                average_precision_micro = multilabel_average_precision(score, labels, num_labels=19, average="micro")
                #average_precision_macro = multilabel_average_precision(score, labels, num_labels=19, average="macro")
                #f1_micro = multilabel_f1_score(score, labels, num_labels=19, average="micro")
                #f1_macro = multilabel_f1_score(score, labels, num_labels=19, average="macro")


                score_time = time.time()-end-data_time-train_time
                
                # print statistics
                running_loss += loss.item()
                running_acc += average_precision_micro
                batch_time = time.time() - end
                end = time.time()        
                sum_bt += batch_time
                sum_dt += data_time
                sum_tt += train_time
                sum_st += score_time
                
                if i % 20 == 19:    # print every 20 mini-batches
    
                    print('[%d, %5d] loss: %.4f mAP_micro: %.4f batch_time: %.4f data_time: %.4f train_time: %.4f score_time: %.4f' %
                          (epoch + 1, i + 1, running_loss / 20, running_acc / 20, sum_bt/20, sum_dt/20, sum_tt/20, sum_st/20))
                    
                    #train_iter =  i*args.batch_size / len(train_dataset)
                    #tb_writer.add_scalar('train_loss', running_loss/20, global_step=(epoch+1+train_iter) )
                    running_loss_epoch = running_loss/20
                    running_acc_epoch = running_acc/20
                    
                    running_loss = 0.0
                    running_acc = 0.0
                    sum_bt = 0.0
                    sum_dt = 0.0
                    sum_tt = 0.0
                    sum_st = 0.0
    
            if epoch % 10 == 9: # evaluate every 10 epochs to save time
                running_loss_val = 0.0
                running_acc_val = 0.0
                count_val = 0
                net.eval()
                with torch.no_grad():
                    for j, data in enumerate(val_loader, 0):
    
                        if args.mode==['s1']:
                            images = data[0][:,12:,:,:] # VV,VH
                            inputs_val, labels_val = images.cuda(), data[1].cuda()
                        elif args.mode==['s2']:                
                            # pad to 13 dimension
                            b_zeros = torch.zeros((data[0].shape[0],1,data[0].shape[2],data[0].shape[3]),dtype=torch.float32)
                            images = torch.cat((data[0][:,:10,:,:],b_zeros,data[0][:,10:12,:,:]),dim=1)
                            inputs_val, labels_val = images.cuda(), data[1].cuda()
                        elif args.mode==['s1','s2']:    
                            images1 = data[0][:,12:,:,:] # VV,VH
                            # pad to 13 dimension
                            b_zeros = torch.zeros((data[0].shape[0],1,data[0].shape[2],data[0].shape[3]),dtype=torch.float32)
                            images2 = torch.cat((data[0][:,:10,:,:],b_zeros,data[0][:,10:12,:,:]),dim=1)
                            inputs1_val, inputs2_val, labels_val = images1.cuda(), images2.cuda(), data[1].cuda()
    
                        if args.mode==['s1','s2']:
                            outputs_val = net(inputs1_val,inputs2_val)
                        else:
                            outputs_val = net(inputs_val)
                            
                        loss_val = criterion(outputs_val, labels_val.long())
                        score = torch.sigmoid(outputs_val).detach()
                        average_precision_micro = multilabel_average_precision(score, labels_val, num_labels=19, average="micro")
                        #average_precision_macro = multilabel_average_precision(score, labels, num_labels=19, average="macro")
                        #f1_micro = multilabel_f1_score(score, labels, num_labels=19, average="micro")
                        #f1_macro = multilabel_f1_score(score, labels, num_labels=19, average="macro")
    
    
                        count_val += 1
                        running_loss_val += loss_val.item()
                        running_acc_val += average_precision_micro        
    
                print('Epoch %d val_loss: %.4f val_mAP_micro: %.4f time: %s seconds.' % (epoch+1, running_loss_val/count_val, running_acc_val/count_val, time.time()-start_time))
    
                if args.rank == 0:
                    losses = {'train': running_loss_epoch,
                              'val': running_loss_val/count_val}
                    accs = {'train': running_acc_epoch,
                            'val': running_acc_val/count_val}        
                    tb_writer.add_scalars('loss', losses, global_step=epoch+1, walltime=None)
                    tb_writer.add_scalars('acc', accs, global_step=epoch+1, walltime=None)
            
                
                
            if args.rank==0 and epoch % 10 == 9:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'loss':loss,
                            }, os.path.join(checkpoints_dir,'checkpoint_{:04d}.pth.tar'.format(epoch)))
            
        #if args.rank==0:
        #    torch.save(net.state_dict(), save_path)
            
        print('Training finished. in %s seconds.' % (time.time()-st_time))



if __name__ == "__main__":
    main()
