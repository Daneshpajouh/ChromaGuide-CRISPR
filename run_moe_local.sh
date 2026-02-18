#!/bin/bash
# Run Novel MoE Architecture on Mac Studio (MPS)
echo "=== CRISPRO-MAMBA-X MoE (Novelty) Local Training ==="
echo "Device: MPS (Apple Silicon)"

# Use 50k subset for reasonable local training time
# Enable --moe flag
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python3 src/train.py --subset_50k --moe
