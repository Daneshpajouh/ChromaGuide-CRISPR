#!/bin/bash
#SBATCH --job-name=debug_mamba_h100
#SBATCH --account=def-kwiese
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

# Load Modules
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2

# Setup Env
export TRITON_CACHE_DIR=~/projects/def-kwiese/amird/tmp/triton_cache
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Activate Venv
source ~/projects/def-kwiese/amird/mamba_h100/bin/activate

# Run Diagnostic
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X || exit 1
python diagnose_h100_cuda.py
