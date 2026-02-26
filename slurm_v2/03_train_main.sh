#!/bin/bash
#SBATCH --job-name=chromaguide-train
#SBATCH --account=def-kwiese
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/train_%A_%a.out
#SBATCH --error=results/logs/train_%A_%a.err
#SBATCH --array=0-14  # 5 backbones × 3 seeds = 15 jobs

# ============================================================
# ChromaGuide: Main Training (Array Job)
# SFU Fir Cluster - 1× A100-80GB per job
#
# Array mapping:
#   0-2:   CNN-GRU (seeds 42, 123, 456)
#   3-5:   DNABERT-2 (seeds 42, 123, 456)
#   6-8:   Nucleotide Transformer (seeds 42, 123, 456)
#   9-11:  Caduceus (seeds 42, 123, 456)
#   12-14: Evo (seeds 42, 123, 456)
# ============================================================

echo "============================================"
echo "ChromaGuide Training"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "============================================"

module load python/3.11
module load cuda/12.2

source $HOME/chromaguide_env/bin/activate
cd $HOME/chromaguide

mkdir -p results/logs results/checkpoints

# Backbone and seed mapping
BACKBONES=(cnn_gru cnn_gru cnn_gru dnabert2 dnabert2 dnabert2 caduceus caduceus caduceus caduceus caduceus caduceus evo evo evo)
SEEDS=(42 123 456 42 123 456 42 123 456 42 123 456 42 123 456)

# Fix array: proper 5 backbones
BACKBONE_NAMES=(cnn_gru dnabert2 nucleotide_transformer caduceus evo)
SEED_VALUES=(42 123 456)

BACKBONE_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

BACKBONE=${BACKBONE_NAMES[$BACKBONE_IDX]}
SEED=${SEED_VALUES[$SEED_IDX]}

echo "Backbone: $BACKBONE, Seed: $SEED"

# Train on Split A (primary)
python -m chromaguide.cli train \
    --backbone $BACKBONE \
    --seed $SEED \
    --split A \
    --wandb \
    2>&1 | tee results/logs/train_${BACKBONE}_seed${SEED}.log

# Also evaluate on Splits B and C
for SPLIT in B C; do
    echo "--- Evaluating on Split $SPLIT ---"
    CKPT="results/checkpoints/${BACKBONE}_splitA_best.pt"
    if [ -f "$CKPT" ]; then
        python -m chromaguide.cli evaluate \
            --checkpoint $CKPT \
            --split $SPLIT \
            2>&1 | tee -a results/logs/train_${BACKBONE}_seed${SEED}.log
    fi
done

echo "Training complete: $(date)"
