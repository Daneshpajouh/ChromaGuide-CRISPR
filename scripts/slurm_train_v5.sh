#!/bin/bash
#SBATCH --job-name=train_cg_v5
#SBATCH --account=def-cloze
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/train_v5_%j.out
#SBATCH --error=logs/train_v5_%j.err

module load python/3.10
module load scipy-stack
source ~/cg_env/bin/activate

# Paths
INPUT="/home/amird/chromaguide_experiments/data/real/processed/enriched_multimodal.h5"
GOLD="/home/amird/chromaguide_experiments/test_set_GOLD.csv"
OUTPUT="results/chromaguide_gold/v5_metrics.json"

export TORCH_COMPILE_DISABLE=1

echo "Starting training V5 at $(date)"
python scripts/train_on_real_data_v5.py \
    --backbone dnabert2 \
    --fusion gate \
    --lr 2e-5 \
    --epochs 20 \
    --batch_size 64 \
    --input "$INPUT" \
    --gold "$GOLD" \
    --output_name "$OUTPUT"

echo "Finished training V5 at $(date)"
