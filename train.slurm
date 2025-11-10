#!/bin/bash
#SBATCH --job-name=fsabau/test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=hai_1031


# Activate environment
source /p/project1/hai_1031/sabau1/gt-substrate-predictor/.venv/bin/activate

# For debugging W&B on clusters:
export WANDB_DIR=$SLURM_TMPDIR
export WANDB_CACHE_DIR=$SLURM_TMPDIR
export WANDB_START_METHOD=thread  # avoids multiprocessing deadlocks

# Run your training
python src/train.py

