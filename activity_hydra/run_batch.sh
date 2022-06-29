#!/bin/bash

#SBATCH --job-name=ptg-activity

#SBATCH --partition=priority
#SBATCH --gres=gpu:rtx6000:2
#SBATCH --cpus-per-task=4

#SBATCH --account=ptg
#SBATCH --mem=16000
#SBATCH --signal=USR1@60
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sri.hegde@kitware.com

#SBATCH --output=./outfiles/ptg_activity.out.%j
#SBATCH --error=./outfiles/ptg_activity.err.%j

conda activate myenv

#HYDRA_FULL_ERROR=1 python train.py trainer.gpus=1
HYDRA_FULL_ERROR=1 python train.py trainer.gpus=4 +trainer.strategy=ddp