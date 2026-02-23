#!/bin/bash
# SLURM job script for FIR - V10 Ablation Modality Study
# Alliance Canada standard: module load + virtualenv in SLURM_TMPDIR

#SBATCH --job-name=v10_ablation_m
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/v10_ablation_modality_%j.out
#SBATCH --error=logs/v10_ablation_modality_%j.err
#SBATCH --mem=32G

set -euo pipefail

module load python/3.11.5
module load cuda/12.2

export VENV_DIR=$SLURM_TMPDIR/env
virtualenv --no-download $VENV_DIR
source $VENV_DIR/bin/activate

pip install torch torchvision --no-index
pip install transformers scipy scikit-learn pandas numpy h5py --no-index

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=~/.cache/torch

cd ~/chromaguide
python -u scripts/run_ablation_modality.py \
    --train_data data/processed/split_a_train.npz \
    --val_data data/processed/split_a_gene_held_out_val.npz \
    --test_data data/processed/split_a_gene_held_out_test.npz \
    --modalities "sequence_only,epigenetic_only,multimodal" \
    --output_path results/ablation_modality_results.json

echo "✓ Modality ablation complete"
