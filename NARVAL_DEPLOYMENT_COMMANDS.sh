#!/bin/bash
# NARVAL DEPLOYMENT SCRIPT
# Copy and paste these commands directly into your Narval terminal
# This avoids SSH timeout issues

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          NARVAL DEPLOYMENT - DNABERT-2 FIX + FOCAL LOSS       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Navigate to repository
echo "Step 1: Navigate to repository..."
cd ~/chromaguide_experiments || exit 1
echo "✅ In directory: $(pwd)"
echo ""

# Step 2: Update from GitHub (if network allows)
echo "Step 2: Attempting git pull..."
if git pull origin main 2>&1 | head -5; then
    echo "✅ Git pull successful"
else
    echo "⚠️  Git pull failed - will use manually transferred files"
fi
echo ""

# Step 3: Activate environment
echo "Step 3: Activating Python environment..."
source ~/env_chromaguide/bin/activate
export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH
echo "✅ Environment activated"
echo ""

# Step 4: Quick validation of DNABERT-2 fix (1 minute, no GPU)
echo "Step 4: Testing DNABERT-2 fix (quick validation)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if timeout 120 python test_dnabert2_fix.py; then
    echo "✅ DNABERT-2 fix verified successfully!"
else
    echo "❌ DNABERT-2 validation failed - check sequence_encoder.py"
    exit 1
fi
echo ""

# Step 5: Submit multimodal training job (50 epochs, ~2 hours)
echo "Step 5: Submitting multimodal training job..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MULTI_JOB=$(sbatch scripts/slurm_multimodal_dnabert2_fixed.sh | awk '{print $NF}')
echo "✅ Multimodal job submitted: $MULTI_JOB"
echo "   Expected runtime: 2 hours"
echo "   Command: sbatch scripts/slurm_multimodal_dnabert2_fixed.sh"
echo ""

# Step 6: Submit off-target training job (200 epochs, ~3 hours)
echo "Step 6: Submitting off-target training job..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
OFF_JOB=$(sbatch scripts/slurm_off_target_focal.sh | awk '{print $NF}')
echo "✅ Off-target job submitted: $OFF_JOB"
echo "   Expected runtime: 3 hours"
echo "   Command: sbatch scripts/slurm_off_target_focal.sh"
echo ""

# Step 7: Check job status
echo "Step 7: Job submission summary..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Multimodal Job ID: $MULTI_JOB"
echo "Off-target Job ID: $OFF_JOB"
echo ""

echo "Monitor jobs with:"
echo "  squeue -u amird"
echo "  squeue -j $MULTI_JOB"
echo "  squeue -j $OFF_JOB"
echo ""

echo "Watch multimodal progress (once started):"
echo "  tail -f slurm_logs/multimodal_dnabert2_fixed_*.out"
echo ""

echo "Watch off-target progress (once started):"
echo "  tail -f slurm_logs/off_target_focal_*.out"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║✅ DEPLOYMENT COMPLETE - Both jobs submitted!                  ║"
echo "║                                                               ║"
echo "║Expected Results (in ~5 hours):                                ║"
echo "║  Multimodal: 0.88-0.92 Rho (Target: 0.911)                    ║"
echo "║  Off-target: 0.82-0.88 AUROC (Target: 0.99)                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
