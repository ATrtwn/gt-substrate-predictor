#!/bin/bash
#SBATCH --job-name=boltz
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/boltz_%j.out
#SBATCH --error=logs/boltz_%j.err
#SBATCH --account=hai_1031

set -e
set -x

mkdir -p logs
NUMBA_CACHE_PATH="/p/scratch/hai_1031/numba_cache_$(whoami)"
mkdir -p "$NUMBA_CACHE_PATH"
# Create a general folder for those math libraries on scratch
GEN_CACHE_PATH="/p/scratch/hai_1031/gen_cache_$(whoami)"
mkdir -p "$GEN_CACHE_PATH"

export BOLTZ_INPUT=/p/project1/hai_1031/gordin1/boltz_input
export BOLTZ_OUTPUTS=/p/project1/hai_1031/gordin1/boltz_output
export BOLTZ_SIF=/p/project1/hai_1031/gordin1/boltz.sif

for yaml_file in "$BOLTZ_INPUT"/*.yaml; do
    filename=$(basename "$yaml_file")
    
    # 1. Get the base name (e.g., config1)
    base_name="${filename%.yaml}"
    
    # 2. Match the actual Boltz output folder name (boltz_results_config1)
    output_folder="boltz_results_$base_name"

    # --- UPDATED SKIP LOGIC ---
    if [ -d "$BOLTZ_OUTPUTS/$output_folder" ]; then
        echo "Skipping $filename: Folder $output_folder already exists."
        continue
    fi
    # --------------------------

    apptainer exec --cleanenv --nv \
        --env NUMBA_CACHE_DIR=/numba_cache \
        --env XDG_CACHE_HOME=/gen_cache \
        --bind "$NUMBA_CACHE_PATH":/numba_cache \
        --bind "$GEN_CACHE_PATH":/gen_cache \
        --bind /p/scratch/hai_1031/.boltz_cache:/boltz_cache \
        --bind "$BOLTZ_INPUT":/inputs \
        --bind /p/project1/hai_1031/gordin1/msa:/msa \
        --bind "$BOLTZ_OUTPUTS":/outputs \
        --bind /p/scratch/hai_1031:/scratch \
        "$BOLTZ_SIF" \
        boltz predict "/inputs/$filename" \
            --cache /boltz_cache \
            --diffusion_samples 5 \
            --out_dir /outputs \
            --override
done