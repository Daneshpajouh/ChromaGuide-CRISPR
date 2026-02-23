#!/bin/bash
# SLURM job script for FIR - V10 Multimodal On-Target Training
# Login: fir.alliancecan.ca
# Request: 1x node with H100 GPU
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
#SBATCH --mail-user=amirdamird@sfu.ca
#SBATCH --mem=64G

set -euo pipefail

echo "=== FIR V10 MULTIMODAL ON-TARGET TRAINING ==="
echo "Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS"
echo ""

# Activate conda environment
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate chromaguide
else
    source ~/chromaguide_venv/bin/activate 2>/dev/null || true
fi

# Set offline mode for Hugging Face (use cached model)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=~/.cache/torch

# Navigate to project
cd ~/chromaguide

echo "Current directory: $(pwd)"
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

# Ensure data directory exists
mkdir -p data/processed

echo "Starting V10 multimodal on-target training..."
echo "Architecture: DNABERT-2 (768-dim) + Per-mark gating (3x256-dim) + Beta head"
echo "Target: Spearman rho >= 0.911 (Split-A gene-held-out)"
echo ""

python3 -u scripts/train_on_real_data_v10.py \
    --backbone dnabert2 \
    --split A \
    --epochs 200 \
    --batch_size 128 \
    --lr 5e-4 \
    --patience 20 \
    --n_seeds 5 \
    --output_prefix v10_multimodal

echo ""
echo "=== Training complete ==="
echo "Results saved to models/ and logs/"
