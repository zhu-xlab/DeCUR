import torch
from torchvision import models

### SAR-optical 
## ssl4eo rn50
pretrained = '/p/oldscratch/hai_ssl4eo/decur/src/pretrain/checkpoints/late_fusion_2p2/B2B13_bt_decu_rn50_prj8192/checkpoint_0099.pth'

checkpoint = torch.load(pretrained, map_location="cpu")
state_dict = checkpoint['model']

# copy the state dict
state_dict_s1 = state_dict.copy()
for k in list(state_dict_s1.keys()):
    if k.startswith("module.backbone_1."):
        state_dict_s1[k.replace("module.backbone_1.", "")] = state_dict_s1[k]
    del state_dict_s1[k]

state_dict_s2 = state_dict.copy()
for k in list(state_dict_s2.keys()):
    if k.startswith("module.backbone_2."):
        state_dict_s2[k.replace("module.backbone_2.", "")] = state_dict_s2[k]
    del state_dict_s2[k]

# save the new state dict
torch.save(state_dict_s1, 'rn50_ssl4eo-s12_sar_decur_ep100.pth')
torch.save(state_dict_s2, 'rn50_ssl4eo-s12_ms_decur_ep100.pth')

# # test the new state dict
# model1 = models.resnet50()
# model1.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# msg1 = model1.load_state_dict(state_dict_s1, strict=False)
# assert set(msg1.missing_keys) == {"fc.weight", "fc.bias"}

# model2 = models.resnet50()
# model2.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# msg2 = model2.load_state_dict(state_dict_s2, strict=False)
# assert set(msg1.missing_keys) == {"fc.weight", "fc.bias"}


## ssl4eo rn50 rda
pretrained = '/p/oldscratch/hai_ssl4eo/decur/src/pretrain/checkpoints/checkpoints/B2B13_rn50_decur_da_res_norm/checkpoint_0049.pth'

checkpoint = torch.load(pretrained, map_location="cpu")
state_dict = checkpoint['model']

# save joint model
torch.save(state_dict, 'rn50_rda_ssl4eo-s12_joint_decur_ep100.pth')
 
# copy the state dict
state_dict_s1 = state_dict.copy()
for k in list(state_dict_s1.keys()):
    if k.startswith("module.backbone_1."):
        state_dict_s1[k.replace("module.backbone_1", "backbone")] = state_dict_s1[k]
    if ('da1_l3' in k or 'da1_l4' in k):
        state_dict_s1[k.replace("module.", "")] = state_dict_s1[k]
    del state_dict_s1[k]

torch.save(state_dict_s1, 'rn50_rda_ssl4eo-s12_sar_decur_ep100.pth')


state_dict_s2 = state_dict.copy()
for k in list(state_dict_s2.keys()):
    if k.startswith("module.backbone_2."):
        state_dict_s2[k.replace("module.backbone_2", "backbone")] = state_dict_s2[k]
    if ('da2_l3' in k or 'da2_l4' in k):
        state_dict_s2[k.replace("module.", "")] = state_dict_s2[k]
    del state_dict_s2[k]

torch.save(state_dict_s2, 'rn50_rda_ssl4eo-s12_ms_decur_ep100.pth')


## ssl4eo vits16
pretrained = '/p/oldscratch/hai_ssl4eo/decur/src/pretrain/checkpoints/checkpoints/B2B13_vits16_decur_norm/checkpoint_0099.pth'

checkpoint = torch.load(pretrained, map_location="cpu")
state_dict = checkpoint['model']

# save joint model
torch.save(state_dict, 'vits16_ssl4eo-s12_joint_decur_ep100.pth')

# copy the state dict
state_dict_s1 = state_dict.copy()
for k in list(state_dict_s1.keys()):
    if k.startswith("module.backbone_1."):
        state_dict_s1[k.replace("module.backbone_1.", "")] = state_dict_s1[k]
    del state_dict_s1[k]

# save the new state dict
torch.save(state_dict_s1, 'vits16_ssl4eo-s12_sar_decur_ep100.pth')

state_dict_s2 = state_dict.copy()
for k in list(state_dict_s2.keys()):
    if k.startswith("module.backbone_2."):
        state_dict_s2[k.replace("module.backbone_2.", "")] = state_dict_s2[k]
    del state_dict_s2[k]
# save the new state dict
torch.save(state_dict_s2, 'vits16_ssl4eo-s12_ms_decur_ep100.pth')

### similar for RGB-DEM on geonrw and RGB-depth on sunrgbd
