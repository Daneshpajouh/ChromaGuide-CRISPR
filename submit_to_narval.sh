#!/bin/bash
################################################################################
# Submit ChromaGuide V2 training jobs to Narval
# 
# Run this from your SSH session on Narval:
#   ssh daneshpajouh@narval.computecanada.ca
#   bash chromaguide_experiments/submit_jobs.sh
################################################################################

cd ~/chromaguide_experiments

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ChromaGuide V2 - Submitting Training Jobs to Narval           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Submit DeepHF training (primary benchmark, 4×H100, 12h)
echo "📤 Submitting DeepHF training job..."
DEEPHF=$(sbatch scripts/slurm_train_v2_deephf.sh)
DEEPHF_ID=$(echo "$DEEPHF" | grep -oP 'Submitted batch job \K\d+' || echo "UNKNOWN")
echo "✅ DeepHF job ID: $DEEPHF_ID"
echo ""

# Submit CRISPRon training (cross-dataset, 4×H100, 12h, can run parallel)
echo "📤 Submitting CRISPRon training job..."
CRISPRON=$(sbatch scripts/slurm_train_v2_crispron.sh)
CRISPRON_ID=$(echo "$CRISPRON" | grep -oP 'Submitted batch job \K\d+' || echo "UNKNOWN")
echo "✅ CRISPRon job ID: $CRISPRON_ID"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✅ JOBS SUBMITTED                                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Timeline:"
echo "  T+0h:    Jobs submitted and begin execution"
echo "  T+12h:   Training complete (both jobs should finish)"
echo "  T+18h:   Statistical evaluation + SOTA comparison complete"
echo ""
echo "Monitor jobs:"
echo "  squeue -u daneshpajouh"
echo "  squeue -j $DEEPHF_ID  # DeepHF status"
echo "  squeue -j $CRISPRON_ID  # CRISPRon status"
echo ""
echo "View logs (live):"
echo "  tail -f logs/train_deephf_*.out"
echo "  tail -f logs/train_crispron_*.out"
echo ""
echo "Check results (after 12h):"
echo "  ls -lh checkpoints/"
echo "  cat checkpoints/deephf_v2/training_results.json"
echo ""
