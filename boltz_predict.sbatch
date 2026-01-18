#!/bin/bash
#BATCH --job-name=boltz
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
export NUMBA_CACHE_DIR=/p/scratch/hai_1031/.numba_cache
export NUMBA_DISABLE_JIT_CACHING=0

module purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Apptainer-Tools/2025


export BOLTZ_CACHE=/p/scratch/hai_1031/.boltz_cache
export BOLTZ_INPUT=/p/project1/hai_1031/gordin1/boltz_input
export BOLTZ_OUTPUTS=/p/project1/hai_1031/gordin1/boltz_output

for yaml_file in "$BOLTZ_INPUT"/*.yaml; do
  apptainer exec --cleanenv --nv \
    --bind /p/scratch/hai_1031/.boltz_cache:/boltz_cache \
    --bind /p/project1/hai_1031/gordin1/boltz_input:/inputs \
    --bind /p/project1/hai_1031/gordin1/msa:/msa \
    --bind /p/project1/hai_1031/gordin1/boltz_output:/outputs \
    /p/project1/hai_1031/gordin1/boltz.sif \
    boltz predict "$yaml_file" \
      --cache /boltz_cache \
      --out_dir /outputs \
      --override
done