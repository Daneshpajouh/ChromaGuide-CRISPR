#!/bin/bash
#SBATCH --job-name=rag_sota_sprint
#SBATCH --output=logs/sota_sprint_v1.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --mem=512GB

# Environment
source setup_h100_env.sh

# Data Integrity Check
if [ ! -f "test_set_GOLD.csv" ]; then
    echo "CRITICAL: Gold Test Set missing! Aborting."
    exit 1
fi

echo "=== STARTING SOTA MAXIMIZATION SPRINT ==="
echo "Strategy: RankNet Loss + 50/50 Focal Sampling + Contrastive RAG"
echo "Dataset: train_val_set.csv (122k samples)"
echo "Test Set: test_set_GOLD.csv (30k samples - HELD OUT)"

# Run Training
# Scale batch size for 8 GPUs: 64 * 8 = 512 effective
torchrun --nproc_per_node=8 src/train.py \
    --batch_size 64 \
    --epochs 50 \
    --lr 2e-4 \
    --rag \
    --dataset "train_val_set.csv"

# Post-Training Benchmark
echo "=== RUNNING GOLD STANDARD BENCHMARK ==="
python3 src/benchmark/benchmarker_gold.py \
    --checkpoint "checkpoints/best_model.pth" \
    --data "test_set_GOLD.csv"

echo "=== SOTA SPRINT COMPLETE ==="
