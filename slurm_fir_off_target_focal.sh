#!/bin/bash
# SLURM job script for FIR - Off-Target Focal Loss Training
# Login: fir.alliancecan.ca
# Request: 1x node with H100 GPU(s)

#SBATCH --job-name=off_target_focal
#SBATCH --account=def-kwhse
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=h100:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/off_target_focal_%j.out
#SBATCH --error=logs/off_target_focal_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amirdamird@sfu.ca
#SBATCH --mem=128G

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   OFF-TARGET FOCAL LOSS TRAINING JOB STARTED              ║"
echo "║   Cluster: FIR (Alliance Canada)                          ║"
echo "║   Job ID: $SLURM_JOB_ID                                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Time: $(date)"
echo "Hostname: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load required modules
module purge
module load gcc/12
module load cuda/12.1
module load nccl
module load python/3.11

# Setup Python environment
VENV_DIR="$HOME/venvs/chromaguide_${SLURM_JOB_ID}"
mkdir -p "$HOME/venvs"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install/upgrade required packages
echo "Setting up Python environment..."
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --quiet transformers accelerate einops biopython scikit-learn pandas numpy

# Setup HuggingFace cache
if [ -n "${SCRATCH:-}" ]; then
    export HF_HOME="$SCRATCH/hf_cache"
    export TRANSFORMERS_CACHE="$SCRATCH/hf_cache/transformers"
    mkdir -p "$TRANSFORMERS_CACHE"
fi

# Environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PYTHONPATH=$HOME/chromaguide/src:$PYTHONPATH

# Navigate to project
cd $HOME/chromaguide
mkdir -p logs

echo "Current directory: $(pwd)"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Run off-target training with focal loss
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting off-target training with Focal Loss..."
echo "Class imbalance handling: Focal Loss (gamma=2.0, alpha=0.25)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -u scripts/train_off_target_focal.py \
    --epochs 200 \
    --batch_size 512 \
    --lr 0.0005 \
    --seed 42 \
    --device cuda

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   OFF-TARGET TRAINING JOB COMPLETED                        ║"
echo "║   Check logs/off_target_focal_$SLURM_JOB_ID.out for results       ║"
echo "╚════════════════════════════════════════════════════════════╝"
