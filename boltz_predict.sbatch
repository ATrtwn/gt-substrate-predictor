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

GEN_CACHE_PATH="/p/scratch/hai_1031/gen_cache_$(whoami)"
mkdir -p "$GEN_CACHE_PATH"

export BOLTZ_INPUT=/p/project1/hai_1031/gordin1/boltz_input
export BOLTZ_OUTPUTS=/p/project1/hai_1031/gordin1/boltz_output
export BOLTZ_SIF=/p/project1/hai_1031/gordin1/boltz.sif

for yaml_file in "$BOLTZ_INPUT"/*.yaml; do
    filename=$(basename "$yaml_file")
    base_name="${filename%.yaml}"
    output_folder="boltz_results_$base_name"
    full_output_path="$BOLTZ_OUTPUTS/$output_folder"
    prediction_path="$full_output_path/predictions"

    # --- UPDATED SKIP LOGIC ---
    # Check if directory exists AND if the predictions folder is NOT empty
    # [ "$(ls -A "$prediction_path" 2>/dev/null)" ] checks if any files exist in that folder
    if [ -d "$full_output_path" ] && [ -d "$prediction_path" ] && [ "$(ls -A "$prediction_path" 2>/dev/null)" ]; then
        echo "Skipping $filename: Valid predictions already exist in $output_folder."
        continue
    else
        echo "Processing $filename: Output missing or predictions folder is empty."
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