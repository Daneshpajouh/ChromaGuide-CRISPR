#!/bin/bash
# 3-Day Sprint: Causal Logic Run (Day 2)
# Launches training with Disentangled Causal Decoder enabled.
# The model produces 3 outputs: Active, Efficiency, Causal
# Loss includes Independence Constraint.

echo "=== SPRINT: Causal Engine (Day 2) ==="
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Using --subset_50k to fail fast/iteration fast
# Run in Mini/Debug Mode (Light on RAM)
python3 src/train.py \
    --data_path data/mini/crisprofft/mini_crisprofft.txt \
    --batch_size 8 \
    --causal \
    --epochs 5 \
    --lr 1e-4
