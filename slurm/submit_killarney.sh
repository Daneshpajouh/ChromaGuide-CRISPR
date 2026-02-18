#!/bin/bash
#SBATCH --job-name=apex_killarney
#SBATCH --account=aip-amird
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/apex_killarney_%j.out
#SBATCH --error=logs/apex_killarney_%j.err

echo "🍀 Launching CRISPRO-Apex on Killarney (L40S x 4)"
echo "------------------------------------------------"

# 1. Environment Setup
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch numpy pandas scipy tqdm transformers

# 2. Execution
export PYTHONPATH=.
python3 src/train_apex.py \
    --data_path "$HOME/projects/def-amird/merged_crispr_data.csv" \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --device cuda
