## DeCUR: decoupling common & unique representations for multimodal self-supervision.

![](assets/decur_main_1.pdf)

<p align="center">
  <img width="500" alt="decur main structure" src="assets/decur_main_structure.png">
</p>

PyTorch implementation of [DeCUR](TBD).


### Pretrained models

| Pretrain dataset | Pretrained model |
| :---: | :---: |
| [SSL4EO-S12](https://arxiv.org/abs/2211.07044) | [SAR/MS]() |
| [GeoNRW](https://ieee-dataport.org/open-access/geonrw) | [RGB/DEM]() |
| [SUNRGBD](https://rgbd.cs.princeton.edu/) | [RGB/HHA]() |

** Download link to be updated. **


### DeCUR Pretraining

Customize your multimodal dataset and your preferred model backbone in `src/datasets/`, `src/models/` and `src/pretrain_mm.py`, and run 

```
python pretrain_mm.py \
--dataset YOUR_DATASET \
--method PRETRAIN_METHOD \
--data1 /path/to/modality1 \
--data2 /path/to/modality2 \
--mode MODAL1 MODAL2 
```

Apart from DeCUR, we also support multimodal pretraining with [SimCLR](https://arxiv.org/abs/2002.05709), [CLIP](https://arxiv.org/abs/2103.00020), [BarlowTwins](https://arxiv.org/abs/2103.03230v3) and [VICReg](https://arxiv.org/abs/2105.04906).

If you are using distributed training with slurm, we provide some example job submission scripts in `src/scripts/pretrain`.

### Transfer Learning

Multilabel scene classification with ResNet50 on [BigEarthNet-MM](https://arxiv.org/abs/2105.07921):

```
$ cd src/transfer_classification_BE
$ python linear_BE.py --backbone resnet50 --mode s1 s2 --pretrained /path/to/pretrained_weights
```

Semantic segmentation with [FCN](https://arxiv.org/abs/1411.4038) on [GeoNRW](https://ieee-dataport.org/open-access/geonrw):

```
$ cd src/transfer_segmentation_GEONRW
$ python GeoNRW_MM_FCN_RN50.py --backbone resnet50 --mode RGB DSM mask --pretrained /path/to/pretrained_weights
```

Semantic segmentation with [CMX](https://arxiv.org/abs/2203.04838) on [SUNRGBD](https://rgbd.cs.princeton.edu/) and [NYUDv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html): 

```
Please refer to https://github.com/huaaaliu/RGBX_Semantic_Segmentation.
Simply load the pretrained weights from your pretrained models. 
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### Citation

TBD.
