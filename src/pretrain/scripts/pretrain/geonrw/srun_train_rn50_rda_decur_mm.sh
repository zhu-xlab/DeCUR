#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B3B1_geonrw_train_rn50_rda_decur_%j.out
#SBATCH --error=srun_outputs/B3B1_geonrw_train_rn50_rda_decur_%j.err
#SBATCH --time=10:00:00
#SBATCH --job-name=lf2_bt_rn50
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=booster

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# load required modules
#export PATH=/p/project1/hai_ssl4eo/wang_yi/software/miniconda3/bin:$PATH
#source /p/project1/hai_ssl4eo/wang_yi/software/miniconda3/etc/profile.d/conda.sh
#conda activate decur

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
--dataset GEONRW \
--mode RGB DSM \
--method DeCUR \
--data1 /p/project/hai_ssl4eo/wang_yi/data/geonrw/nrw_dataset_250x250/img_dir/train \
--data2 /p/project/hai_ssl4eo/wang_yi/data/geonrw/nrw_dataset_250x250/dem_dir/train \
--epochs 100 \
--batch-size 64 \
--workers 10 \
--learning-rate-weights 1e-4 \
--learning-rate-biases 1e-4 \
--weight-decay 0.01 \
--lambd 0.0051 \
--projector 8192-8192-8192 \
--print-freq 100 \
--checkpoint-dir /p/project/hai_ssl4eo/wang_yi/decur/DeCUR/src/pretrain/checkpoints/B3B1_geonrw_decur_rn50 \
--backbone resnet50 \
--dist_url $dist_url \
--cos \
--dim_common 6144 \
--rda \
#--resume /p/project/hai_dm4eo/wang_yi/ssl4eo-mm-v3/src/pretrain/checkpoints/geonrw/B3B1_lf2_bt_decu_rn50/checkpoint_0096.pth
