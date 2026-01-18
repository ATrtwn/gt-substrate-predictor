#!/bin/bash
#SBATCH --job-name=boltz
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/boltz_%j.out
#SBATCH --error=logs/boltz_%j.err
#SBATCH --account=hai_1031

set -e
set -x

mkdir -p logs
mkdir -p /p/scratch/hai_1031/.numba_cache

export BOLTZ_CACHE=/p/scratch/hai_1031/.boltz_cache
export BOLTZ_INPUT=/p/project1/hai_1031/gordin1/boltz_input
export BOLTZ_OUTPUTS=/p/project1/hai_1031/gordin1/boltz_output

for yaml_file in "$BOLTZ_INPUT"/*.yaml; do
  apptainer exec --cleanenv --nv \
    --env NUMBA_DISABLE_CACHING=1 \
    --bind /p/scratch/hai_1031/.boltz_cache:/boltz_cache \
    --bind "$BOLTZ_INPUT":/inputs \
    --bind /p/project1/hai_1031/gordin1/msa:/msa \
    --bind "$BOLTZ_OUTPUTS":/outputs \
    --bind /p/scratch/hai_1031:/scratch \
    /p/project1/hai_1031/gordin1/boltz.sif \
    boltz predict "/inputs/$(basename "$yaml_file")" \
      --cache /boltz_cache \
      --out_dir /outputs \
      --override
done