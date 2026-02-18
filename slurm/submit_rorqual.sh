#!/bin/bash
#SBATCH --job-name=apex_rorqual
#SBATCH --account=def-amird
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/apex_rorqual_%j.out
#SBATCH --error=logs/apex_rorqual_%j.err

echo "🐋 Launching CRISPRO-Apex on Rorqual (H100 x 4)"
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
    --epochs 10 \
    --batch_size 512 \
    --lr 5e-4 \
    --device cuda
