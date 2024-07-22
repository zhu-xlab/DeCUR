#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/BE_s2_LC_vits16_decur_1_%j.out
#SBATCH --error=srun_outputs/BE_s2_LC_vits16_decur_1_%j.err
#SBATCH --time=00:20:00
#SBATCH --job-name=BE_LC_s1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=develbooster

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# load required modules
# module load Stages/2022
# module load GCCcore/.11.2.0
# module load Python

# # activate virtual environment
# source /p/project/hai_dm4eo/wang_yi/env2/bin/activate

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# run script as slurm job
srun python -u linear_BE_vit.py \
--lmdb_dir /p/home/jusers/wang36/juwels/project_hai_ssl4eo/wang_yi/data/BigEarthNet/BigEarthNet_LMDB_raw \
--mode s2 \
--checkpoints_dir checkpoints/BE_s2_LC_vits16_decur_1 \
--backbone vits16 \
--train_frac 0.01 \
--batchsize 64 \
--lr 0.1 \
--schedule 60 80 \
--epochs 100 \
--num_workers 10 \
--seed 42 \
--dist_url $dist_url \
--linear \
--pretrained /p/project/hai_ssl4eo/wang_yi/decur/DeCUR/src/pretrain/checkpoints/vits16_ssl4eo-s12_joint_decur_ep100.pth \
#--rda \
