import torch
import torch.nn as nn
import torchvision
import diffdist

class CLIP(nn.Module):

    LARGE_NUMBER = 1e9

    def __init__(self,args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet50':
            self.backbone_1 = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet50(zero_init_residual=True)
            ndim_proj = 2048
        elif args.backbone == 'resnet18':
            self.backbone_1 = torchvision.models.resnet18(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet18(zero_init_residual=True)
            ndim_proj = 512
        elif args.backbone == 'vits16':
            import timm
            self.backbone_1 = timm.create_model('vit_small_patch16_224',pretrained=True)
            self.backbone_2 = timm.create_model('vit_small_patch16_224',pretrained=True)
            ndim_proj = 384
            
        if 'resnet' in args.backbone:
            if args.mode==['s1','s2c']:
                self.backbone_1.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
                self.backbone_2.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)        
            self.backbone_1.fc = nn.Identity()
            self.backbone_2.fc = nn.Identity()
        elif 'vit' in args.backbone:
            if args.mode==['s1','s2c']:
                self.backbone_1.patch_embed.proj = nn.Conv2d(2, 384, kernel_size=(16, 16), stride=(16, 16))
                self.backbone_2.patch_embed.proj = nn.Conv2d(13, 384, kernel_size=(16, 16), stride=(16, 16))
            self.backbone_1.head = nn.Identity()
            self.backbone_2.head = nn.Identity()
                    
        self.projector1 = nn.Sequential(nn.Linear(ndim_proj, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128), nn.BatchNorm1d(128))
        self.projector2 = nn.Sequential(nn.Linear(ndim_proj, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128), nn.BatchNorm1d(128))
            
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