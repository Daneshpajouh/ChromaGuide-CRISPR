#!/bin/bash
# SLURM job script for FIR - Multimodal DNABERT-2 Training
# Login: fir.alliancecan.ca
# Request: 1x node with H100 GPU(s)

#SBATCH --job-name=multimodal_dnabert2
#SBATCH --account=def-kwhse
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=h100:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/multimodal_dnabert2_%j.out
#SBATCH --error=logs/multimodal_dnabert2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amirdamird@sfu.ca
#SBATCH --mem=128G

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   MULTIMODAL DNABERT-2 TRAINING JOB STARTED               ║"
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
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
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

# Quick DNABERT-2 validation
echo "Testing DNABERT-2 fix..."
if python test_dnabert2_fix.py 2>&1 | tail -1 | grep -q SUCCESS; then
    echo "✅ DNABERT-2 fix verified!"
else
    echo "⚠️  DNABERT-2 validation encountered issues (continuing anyway)"
fi
echo ""

# Run multimodal training
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting multimodal training with DNABERT-2 backbone..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -u scripts/train_on_real_data_v2.py \
    --backbone dnabert2 \
    --epochs 50 \
    --batch_size 250 \
    --lr 5e-4 \
    --device cuda \
    --seed 42

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   MULTIMODAL TRAINING JOB COMPLETED                       ║"
echo "║   Check logs/multimodal_dnabert2_$SLURM_JOB_ID.out for results    ║"
echo "╚════════════════════════════════════════════════════════════╝"
