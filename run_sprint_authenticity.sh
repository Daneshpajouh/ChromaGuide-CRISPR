#!/bin/bash
# 3-Day Sprint: Authenticity Run
# Launches standard training but with Biophysical Gating logic enabled.
# The Dataset loader now auto-computes features.
# The Model now gates via physics.

echo "=== SPRINT: Authenticity Engine ==="
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Using --subset_50k to fail fast/iteration fast
# Using --production settings for everything else
# Run in Mini/Debug Mode (Light on RAM)
python3 src/train.py \
    --data_path data/mini/crisprofft/mini_crisprofft.txt \
    --batch_size 8 \
    --epochs 5 \
    --lr 1e-4
