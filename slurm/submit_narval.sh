#!/bin/bash
#SBATCH --job-name=apex_narval
#SBATCH --account=def-amird
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/apex_narval_%j.out
#SBATCH --error=logs/apex_narval_%j.err

echo "🚀 Launching CRISPRO-Apex on Narval (A100 x 4)"
echo "------------------------------------------------"

# 1. Environment Setup
module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch numpy pandas scipy tqdm transformers

# 2. Data Preparation
# Assuming data is in ~/projects/def-amird/dataset/
DATA_PATH="$HOME/projects/def-amird/merged_crispr_data.csv"

# 3. Execution (Distributed Data Parallel ready)
export PYTHONPATH=.
python3 src/train_apex.py \
    --data_path $DATA_PATH \
    --epochs 10 \
    --batch_size 256 \
    --lr 2e-4 \
    --device cuda
