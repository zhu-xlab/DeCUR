#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/geonrw/B3B1_train_rn50_simclr_%j.out
#SBATCH --error=srun_outputs/geonrw/B3B1_train_rn50_simclr_%j.err
#SBATCH --time=05:00:00
#SBATCH --job-name=lf_simclr_rn50
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=booster

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
srun python -u pretrain_mm.py \
--dataset GEONRW \
--mode RGB DSM \
--method SimCLR \
--data1 /p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/img_dir/train \
--data2 /p/project/hai_ssl4eo/wang_yi/data/nrw_dataset_250x250/dem_dir/train \
--epochs 100 \
--batch-size 64 \
--workers 10 \
--learning-rate-weights 0.2 \
--learning-rate-biases 0.0048 \
--weight-decay 1e-6 \
--print-freq 100 \
--checkpoint-dir /p/project/hai_dm4eo/wang_yi/ssl4eo-mm-v3/src/pretrain/checkpoints/geonrw/B3B1_simclr_rn50 \
--backbone resnet50 \
--dist_url $dist_url \
--cos \
#--resume /p/project/hai_dm4eo/wang_yi/ssl4eo-mm-v3/src/pretrain/checkpoints/late_fusion_p2/B2B13_bt_rn50/checkpoint_0040.pth
