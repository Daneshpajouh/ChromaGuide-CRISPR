#!/bin/bash
# Job 4: The Grand Unification (Causal + Quantum + Topo)
# This job runs the full theoretical stack.

# Enabled Causal, Quantum, Topo
export PYTHONPATH=$PYTHONPATH:.
# Run in Mini/Debug Mode (Light on RAM)
python3 src/train.py \
    --data_path data/mini/crisprofft/mini_crisprofft.txt \
    --batch_size 8 \
    --causal \
    --quantum \
    --topo \
    --epochs 5 \
    --lr 1e-4
