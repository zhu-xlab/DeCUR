import torch

### MITB2
ckpt_path = 'mitb2_sunrgbd_rgb_hha_decur_ep200.pth'
ckpt = torch.load(ckpt_path,map_location='cpu')

state_dict = ckpt['model']
state_dict1 = state_dict.copy()
state_dict2 = state_dict.copy()
for k in list(state_dict.keys()):
    if k.startswith('module.backbone_1') and not k.startswith('module.backbone_1.head'):
        #state_dict[k[len("module.backbone_1."):]] = state_dict[k]
        state_dict1[k[len("module.backbone_1."):]] = state_dict1[k]
    del state_dict1[k]
    if k.startswith('module.backbone_2') and not k.startswith('module.backbone_2.head'):
        #state_dict[k[len("module.backbone_1."):]] = state_dict[k]
        state_dict2[k[len("module.backbone_2."):]] = state_dict2[k]
    del state_dict2[k]    
    
    
torch.save(state_dict1,'decur_mitb2_rgb.pth')
torch.save(state_dict2,'decur_mitb2_hha.pth')



'''
### MITB5
ckpt_path = 'mitb2_sunrgbd_rgb_hha_decur_ep200.pth.pth'
ckpt = torch.load(ckpt_path,map_location='cpu')

state_dict = ckpt['model']
state_dict1 = state_dict.copy()
state_dict2 = state_dict.copy()
for k in list(state_dict.keys()):
    if k.startswith('module.backbone_1') and not k.startswith('module.backbone_1.head'):
        #state_dict[k[len("module.backbone_1."):]] = state_dict[k]
        state_dict1[k[len("module.backbone_1."):]] = state_dict1[k]
    del state_dict1[k]
    if k.startswith('module.backbone_2') and not k.startswith('module.backbone_2.head'):
        #state_dict[k[len("module.backbone_1."):]] = state_dict[k]
        state_dict2[k[len("module.backbone_2."):]] = state_dict2[k]
    del state_dict2[k]    
    
    
torch.save(state_dict1,'decur_mitb5_rgb.pth')
torch.save(state_dict2,'decur_mitb5_hha.pth')
'''