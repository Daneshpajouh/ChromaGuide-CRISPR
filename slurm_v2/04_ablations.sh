#!/bin/bash
#SBATCH --job-name=chromaguide-ablation
#SBATCH --account=def-kwiese
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/ablation_%A_%a.out
#SBATCH --error=results/logs/ablation_%A_%a.err
#SBATCH --array=0-11

# ============================================================
# ChromaGuide: Ablation Studies (Array Job)
#
# Array mapping:
# Epigenomic ablations (CNN-GRU backbone, seed 42):
#   0: no epigenomic (sequence only)
#   1: accessibility only (DNase)
#   2: histone only (H3K4me3 + H3K27ac)
#   3: full (all 3 tracks)
#   4: mismatched cell line
#
# Fusion ablations (CNN-GRU backbone, seed 42):
#   5: concat_mlp
#   6: gated_attention
#   7: cross_attention
#   8: moe
#
# Special ablations:
#   9:  no modality dropout
#   10: no calibration loss
#   11: MSE instead of Beta NLL
# ============================================================

echo "============================================"
echo "ChromaGuide Ablation Study"
echo "Job ID: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"
echo "============================================"

module load python/3.11
module load cuda/12.2

source $HOME/chromaguide_env/bin/activate
cd $HOME/chromaguide

mkdir -p results/logs

TASK_ID=$SLURM_ARRAY_TASK_ID

case $TASK_ID in
    0)
        NAME="seq_only"
        OVERRIDES="model.modality_dropout.prob=1.0"  # Always mask epigenomic
        ;;
    1)
        NAME="accessibility_only"
        OVERRIDES="data.epigenomic.tracks=[DNase]"
        ;;
    2)
        NAME="histone_only"
        OVERRIDES="data.epigenomic.tracks=[H3K4me3,H3K27ac]"
        ;;
    3)
        NAME="full_epigenomic"
        OVERRIDES=""
        ;;
    4)
        NAME="mismatched_cellline"
        OVERRIDES=""  # Handled in data loading
        ;;
    5)
        NAME="fusion_concat_mlp"
        OVERRIDES="model.fusion.type=concat_mlp"
        ;;
    6)
        NAME="fusion_gated_attention"
        OVERRIDES="model.fusion.type=gated_attention"
        ;;
    7)
        NAME="fusion_cross_attention"
        OVERRIDES="model.fusion.type=cross_attention"
        ;;
    8)
        NAME="fusion_moe"
        OVERRIDES="model.fusion.type=moe"
        ;;
    9)
        NAME="no_modality_dropout"
        OVERRIDES="model.modality_dropout.enabled=false"
        ;;
    10)
        NAME="no_cal_loss"
        OVERRIDES="training.loss.lambda_cal=0.0"
        ;;
    11)
        NAME="mse_loss"
        OVERRIDES="training.loss.primary=mse"
        ;;
esac

echo "Ablation: $NAME"
echo "Overrides: $OVERRIDES"

python -m chromaguide.cli train \
    --backbone cnn_gru \
    --seed 42 \
    --split A \
    --wandb \
    $OVERRIDES \
    2>&1 | tee results/logs/ablation_${NAME}.log

echo "Ablation $NAME complete: $(date)"
