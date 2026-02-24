#!/bin/bash
# SLURM job script for FIR - V10 Off-Target Classification Training
# Alliance Canada standard: module load + virtualenv in SLURM_TMPDIR
# Purpose: Train DNABERT-2 + linear classifier for off-target prediction
# Target: AUROC >= 0.99

#SBATCH --job-name=v10_offtarget
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/v10_offtarget_%j.out
#SBATCH --error=logs/v10_offtarget_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=64G

set -euo pipefail

echo "=== FIR V10 OFF-TARGET TRAINING ==="
echo "Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS"
echo ""

# Initialize module system (required on Alliance Canada compute nodes)
source /etc/profile.d/modules.sh

# Load required modules (Alliance Canada standard)
module load python/3.11.5
module load cuda/12.2

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

# Run off-target training
echo "Starting off-target training..."
python -u scripts/train_off_target_v10.py \
    --data_path ~/chromaguide/data/raw/crisprofft/CRISPRoffT_all_targets.txt \
    --epochs 8 \
    --batch_size 128

echo ""
echo "✓ Off-target training complete at $(date)"
