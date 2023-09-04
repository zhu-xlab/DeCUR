#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/sunrgbd/B3B3_train_mitb5_decur_%j.out
#SBATCH --error=srun_outputs/sunrgbd/B3B3_train_mitb5_decur_%j.err
#SBATCH --time=10:00:00
#SBATCH --job-name=decur_rgbx
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
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# run script as slurm job
srun python -u pretrain_mm.py \
--dataset SUNRGBD \
--mode rgb hha \
--method DeCUR \
--data1 /p/project/hai_ssl4eo/wang_yi/data/sun_rgbd/sunrgbd_image \
--data2 /p/project/hai_ssl4eo/wang_yi/data/sun_rgbd/sunrgbd_hha \
--epochs 200 \
--batch-size 32 \
--workers 10 \
--learning-rate-weights 1e-4 \
--lambd 0.0051 \
--projector 8192-8192-8192 \
--print-freq 100 \
--checkpoint-dir /p/project/hai_dm4eo/wang_yi/decur/src/pretrain/checkpoints/sunrgbd/B3B3_mitb5_decur \
--backbone mit_b5 \
--dist_url $dist_url \
--cos \
--dim_common 6144 \
#--pretrained /p/project/hai_dm4eo/wang_yi/decur/src/pretrain/pretrained/mit_b5.pth \
#--resume /p/project/hai_dm4eo/wang_yi/decur/src/pretrain/checkpoints/sunrgbd/B3B3_train_decur_prj8192_ep200_bs16_rgbx_mitb2/checkpoint_0159.pth
