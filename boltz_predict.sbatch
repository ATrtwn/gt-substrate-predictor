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
# Create a specific directory for Numba on scratch
NUMBA_CACHE_PATH="/p/scratch/hai_1031/numba_cache_$(whoami)"
mkdir -p "$NUMBA_CACHE_PATH"

export BOLTZ_INPUT=/p/project1/hai_1031/gordin1/boltz_input
export BOLTZ_OUTPUTS=/p/project1/hai_1031/gordin1/boltz_output
export BOLTZ_SIF=/p/project1/hai_1031/gordin1/boltz.sif

for yaml_file in "$BOLTZ_INPUT"/*.yaml; do
    filename=$(basename "$yaml_file")
    
    apptainer exec --cleanenv --nv \
        --env NUMBA_CACHE_DIR=/numba_cache \
        --bind "$NUMBA_CACHE_PATH":/numba_cache \
        --bind /p/scratch/hai_1031/.boltz_cache:/boltz_cache \
        --bind "$BOLTZ_INPUT":/inputs \
        --bind /p/project1/hai_1031/gordin1/msa:/msa \
        --bind "$BOLTZ_OUTPUTS":/outputs \
        --bind /p/scratch/hai_1031:/scratch \
        "$BOLTZ_SIF" \
        boltz predict "/inputs/$filename" \
            --cache /boltz_cache \
            --out_dir /outputs \
            --override
done