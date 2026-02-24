#!/bin/bash
# SLURM job script for FIR - V10 Multimodal On-Target Training
# Alliance Canada standard: module load + virtualenv in SLURM_TMPDIR
# Purpose: Train DNABERT-2 + per-mark gating + Beta regression for efficacy prediction
# Target: Spearman rho >= 0.911 on Split-A gene-held-out validation

#SBATCH --job-name=v10_multimodal
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/v10_multimodal_%j.out
#SBATCH --error=logs/v10_multimodal_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=64G

set -euo pipefail

echo "=== FIR V10 MULTIMODAL ON-TARGET TRAINING ==="
echo "Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS"
echo ""

# Initialize module system (required on Alliance Canada compute nodes)
# Try multiple possible module paths on FIR
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
else
    # Try to load modules directly (fallback for compute nodes)
    if command -v module >/dev/null 2>&1; then
        echo "Module command available"
    else
        echo "⚠️  Module system not available - using system Python"
    fi
fi

# Load required modules (Alliance Canada standard)
if command -v module >/dev/null 2>&1; then
    module load python/3.11.5 2>/dev/null || echo "⚠️  python/3.11.5 not available"
    module load cuda/12.2 2>/dev/null || echo "⚠️  cuda/12.2 not available"
else
    echo "⚠️  Module system not available - proceeding with system paths"
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA: $(which nvcc)"
echo ""

# Create virtual environment in SLURM_TMPDIR (fast SSD)
export VENV_DIR=$SLURM_TMPDIR/env
echo "Creating virtualenv in $VENV_DIR..."
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

echo "Virtual environment activated"
echo "Python: $(which python)"
echo ""

# Install packages from pre-built wheels (no compilation)
echo "Installing PyTorch + dependencies..."
pip install torch torchvision --no-index
pip install transformers scipy scikit-learn pandas numpy h5py --no-index

echo "Packages installed"
echo ""

# Cache paths for Hugging Face models (will download on first run)
# NOTE: TRANSFORMERS_OFFLINE not set on first run - allows DNABERT-2 download
export TORCH_HOME=~/.cache/torch

# Navigate to project
cd ~/chromaguide
echo "Working directory: $(pwd)"
echo ""

# Run training
echo "Starting multimodal training..."
python -u scripts/train_on_real_data_v10.py \
    --split A \
    --epochs 200 \
    --batch_size 128 \
    --lr 5e-4 \
    --n_seeds 5

echo ""
echo "✓ Multimodal training complete at $(date)"
