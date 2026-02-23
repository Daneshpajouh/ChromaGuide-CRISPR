#!/bin/bash
# SLURM job script for FIR - V10 Bootstrap Statistical Testing
# Purpose: Wilcoxon signed-rank tests vs ChromeCRISPR baseline
# Target: p < 0.001, report Cohen's d effect sizes

#SBATCH --job-name=v10_bootstrap
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --time=1:30:00
#SBATCH --output=logs/v10_bootstrap_%j.out
#SBATCH --error=logs/v10_bootstrap_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G

set -euo pipefail

echo "=== Bootstrap Statistical Testing ==="
cd ~/chromaguide

python3 -u scripts/run_bootstrap_testing.py \
    --v10_preds models/v10_multimodal_predictions.pkl \
    --baseline_preds models/chromecrispr_predictions.pkl \
    --test_data data/processed/split_a_gene_held_out_test.npz \
    --n_bootstrap 10000 \
    --output_path results/bootstrap_stats.json

echo "✓ Bootstrap testing complete"
