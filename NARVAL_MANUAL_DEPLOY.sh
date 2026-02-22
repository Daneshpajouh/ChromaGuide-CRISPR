#!/bin/bash
# Manual deployment script for Narval (if SSH fails)
# Run this on Narval directly OR copy files manually via SCP

echo "Step 1: Update source code"
echo "Copy these files to ~/.../chromaguide_experiments/:"
echo "  - src/chromaguide/sequence_encoder.py (DNABERT-2 fix)"
echo "  - scripts/train_on_real_data_v2.py"
echo "  - scripts/slurm_multimodal_dnabert2_fixed.sh"

echo ""
echo "Step 2: Verify DNABERT-2 cache"
ls -la ~/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/*/pytorch_model.bin

echo ""
echo "Step 3: Test DNABERT-2 loading (CPU only - quick test)"
cd ~/chromaguide_experiments
python3 test_dnabert2_fix.py

echo ""
echo "Step 4: Submit training job if test passes"
sbatch scripts/slurm_multimodal_dnabert2_fixed.sh
echo "Check status with: squeue -u amird"

