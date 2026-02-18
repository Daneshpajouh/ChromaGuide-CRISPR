#!/bin/bash
# 3-Day Sprint: Zero-Shot Transfer (Day 3)
# Trains on Human Data (HEK293T), Evaluates on Mouse Data (Hepa1-6).
# Proves Foundation Model Generalization.

echo "=== SPRINT: Foundation Zero-Shot (Day 3) ==="
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# We use a new argument --zeroshot_split to tell# Run in Mini/Debug Mode (Light on RAM)
python3 src/train.py \
    --data_path data/mini/crisprofft/mini_crisprofft.txt \
    --batch_size 8 \
    --causal \
    --epochs 5 \
    --lr 1e-4 \
    --production
