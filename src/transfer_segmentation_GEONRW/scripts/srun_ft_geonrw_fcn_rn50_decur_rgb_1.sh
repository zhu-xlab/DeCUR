#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/geonrw_rgb_FCN_rn50_ft_bt_lf2_decu_1_prj8192_%j.out
#SBATCH --error=srun_outputs/geonrw_rgb_FCN_rn50_ft_bt_lf2_decu_1_prj8192_%j.err
#SBATCH --time=00:20:00
#SBATCH --job-name=geonrw_ft
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=develbooster

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# load required modules
module load Stages/2022
module load GCCcore/.11.2.0
module load Python

# activate virtual environment
source /p/project/hai_dm4eo/wang_yi/env2/bin/activate

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

# run script as slurm job
srun python -u GeoNRW_RGB_FCN_RN50.py \
--data1 /p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/img_dir \
--data2 /p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/dem_dir \
--mask /p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/ann_dir \
--mode RGB DSM mask \
--checkpoints_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-mm-v3/src/transfer_segmentation/checkpoints/geonrw_rgb_FCN_rn50_ft_bt_lf2_decu_1_prj8192 \
--backbone resnet50 \
--train_frac 0.01 \
--batchsize 64 \
--lr 0.0001 \
--epochs 30 \
--num_workers 10 \
--seed 42 \
--dist_url $dist_url \
--pretrained /p/project/hai_dm4eo/wang_yi/ssl4eo-mm-v3/src/pretrain/checkpoints/geonrw/B3B1_lf2_bt_decu_rn50_cd384_prj8192/checkpoint_0099.pth \
#--linear \
#--schedule 60 80 \
#--resume /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/moco_lc/BE_rn50_10_r112/checkpoint_0009.pth.tar
