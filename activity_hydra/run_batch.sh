#!/bin/bash

# This is a slurm utility script for scheduling the training job 
# on SLURM scheduler (for eg. on numenor server). This file could
# be added under "scripts" folder if required (make suitable path
# adjustments in this case).


#SBATCH --job-name=ptg-activity

#SBATCH --partition=priority
#SBATCH --gres=gpu:rtx6000:2
#SBATCH --cpus-per-task=4

#SBATCH --account=ptg
#SBATCH --mem=30000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sri.hegde@kitware.com

#SBATCH --output=./outfiles/ptg_activity.out.%j
#SBATCH --error=./outfiles/ptg_activity.err.%j

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# HYDRA_FULL_ERROR=1 python train.py trainer.gpus=1
HYDRA_FULL_ERROR=1 python train.py trainer.gpus=2 +trainer.strategy=ddp

# Resuming training from checkpoint
# HYDRA_FULL_ERROR=1 python train.py trainer.resume_from_checkpoint="checkpoints/rulstm_ep_24.ckpt" trainer.gpus=2 +trainer.strategy=ddp
