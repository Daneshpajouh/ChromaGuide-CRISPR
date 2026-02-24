#!/bin/bash
# SLURM job script for FIR - V10 Ablation Fusion Study
# Alliance Canada standard: module load + virtualenv in SLURM_TMPDIR

#SBATCH --job-name=v10_ablation_f
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/v10_ablation_fusion_%j.out
#SBATCH --error=logs/v10_ablation_fusion_%j.err
#SBATCH --mem=32G

set -euo pipefail

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

export VENV_DIR=$SLURM_TMPDIR/env
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip install torch torchvision
pip install transformers scipy scikit-learn pandas numpy h5py einops

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=~/.cache/torch

cd ~/chromaguide
python -u scripts/run_ablation_fusion.py \
    --train_data data/processed/split_a_train.npz \
    --val_data data/processed/split_a_gene_held_out_val.npz \
    --test_data data/processed/split_a_gene_held_out_test.npz \
    --fusion_methods "gating,concatenation,cross_attention" \
    --output_path results/ablation_fusion_results.json

echo "✓ Fusion ablation complete"
