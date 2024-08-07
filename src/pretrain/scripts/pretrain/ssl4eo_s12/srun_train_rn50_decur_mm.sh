#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B2B13_train_rn50_decur_%j.out
#SBATCH --error=srun_outputs/B2B13_train_rn50_decur_%j.err
#SBATCH --time=00:30:00
#SBATCH --job-name=decur_rn50
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=develbooster

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# load required modules
# activate virtual environment
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
--dataset SSL4EO \
--mode s1 s2c \
--method DeCUR \
--data1 /p/oldscratch/hai_ssl4eo/wang_yi/data/ssl4eo-s12_0k_251k_sar.lmdb \
--data2 /p/oldscratch/hai_ssl4eo/wang_yi/data/ssl4eo-s12_0k_251k_s2c_norm.lmdb \
--epochs 100 \
--batch-size 64 \
--workers 10 \
--learning-rate-weights 0.05 \
--learning-rate-biases 0.0048 \
--weight-decay 1e-6 \
--lambd 0.0051 \
--projector 8192-8192-8192 \
--print-freq 100 \
--checkpoint-dir /p/project1/hai_ssl4eo/wang_yi/decur/DeCUR/src/pretrain/checkpoints/B2B13_ssl4eo_decur_rn50 \
--backbone resnet50 \
--dist_url $dist_url \
--cos \
--dim_common 7168 \
#--rda \
#--resume /p/project/hai_dm4eo/wang_yi/decur/src/pretrain/checkpoints/late_fusion_2p2/B2B13_bt_decu_rn50_prj8192/checkpoint_0061.pth
