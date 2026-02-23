#!/bin/bash
set -e
cd /Users/studio/Desktop/PhD/Proposal

echo "Waiting for v6 & v7 trainings to complete..."
echo "Monitoring multimodal v6 and off-target v7 CNN..."
echo ""

# Wait for both to complete (check every 2 min, max 6 hours)
for i in {1..180}; do
    MULTIMODAL_RUNNING=$(pgrep -f "train_on_real_data_v6" > /dev/null 2>&1 && echo "yes" || echo "no")
    OFFTARGET_RUNNING=$(pgrep -f "train_off_target_v7_cnn" > /dev/null 2>&1 && echo "yes" || echo "no")

    if [ "$MULTIMODAL_RUNNING" = "no" ] && [ "$OFFTARGET_RUNNING" = "no" ]; then
        echo "✓ Both trainings completed!"
        break
    fi

    if [ $((i % 5)) -eq 0 ]; then
        echo "$(date): Still training... (check $i/180)"
    fi

    sleep 120
done

# Now run STEPS 5-8
echo ""
echo "==========================================================="
echo "STEP 5: Running Evaluation Pipeline"
echo "==========================================================="

# Check if evaluation scripts exist
if [ -f "scripts/evaluate_all_targets.py" ]; then
    echo "Running evaluate_all_targets.py..."
    conda run --no-capture-output -n cg_train python scripts/evaluate_all_targets.py 2>&1 | tee logs/evaluate_all_targets.log || echo "⚠ evaluate_all_targets.py failed"
else
    echo "⚠ evaluate_all_targets.py not found"
fi

if [ -f "scripts/run_bootstrap_testing.py" ]; then
    echo "Running run_bootstrap_testing.py..."
    conda run --no-capture-output -n cg_train python scripts/run_bootstrap_testing.py 2>&1 | tee logs/bootstrap_testing.log || echo "⚠ run_bootstrap_testing.py failed"
else
    echo "⚠ run_bootstrap_testing.py not found"
fi

if [ -f "scripts/calibrate_conformal.py" ]; then
    echo "Running calibrate_conformal.py..."
    conda run --no-capture-output -n cg_train python scripts/calibrate_conformal.py 2>&1 | tee logs/conformal_calibration.log || echo "⚠ calibrate_conformal.py failed"
else
    echo "⚠ calibrate_conformal.py not found"
fi

if [ -f "scripts/run_designer.py" ]; then
    echo "Running run_designer.py..."
    conda run --no-capture-output -n cg_train python scripts/run_designer.py 2>&1 | tee logs/designer_score.log || echo "⚠ run_designer.py failed"
else
    echo "⚠ run_designer.py not found"
fi

echo ""
echo "==========================================================="
echo "STEP 6: Running Ablation Studies"
echo "==========================================================="

if [ -f "scripts/run_ablation_fusion.py" ]; then
    echo "Running fusion ablation..."
    conda run --no-capture-output -n cg_train python scripts/run_ablation_fusion.py 2>&1 | tee logs/ablation_fusion.log || echo "⚠ run_ablation_fusion.py failed"
else
    echo "⚠ run_ablation_fusion.py not found"
fi

if [ -f "scripts/run_ablation_modality.py" ]; then
    echo "Running modality ablation..."
    conda run --no-capture-output -n cg_train python scripts/run_ablation_modality.py 2>&1 | tee logs/ablation_modality.log || echo "⚠ run_ablation_modality.py failed"
else
    echo "⚠ run_ablation_modality.py not found"
fi

echo ""
echo "==========================================================="
echo "STEP 7: Git Commit and Push"
echo "==========================================================="

git add -A 2>/dev/null || true
git commit -m "v6 training results and evaluation: multimodal (d_model=64, cross-attention) + off-target (residual net ensemble)" 2>/dev/null || echo "No changes to commit"
git push origin main 2>/dev/null || echo "Push failed - check network"

echo ""
echo "==========================================================="
echo "STEP 8: Final Metrics vs Targets"
echo "==========================================================="

echo ""
echo "Target Metrics:"
echo "  Multimodal: Spearman rho >= 0.911 (Split-A gene-held-out)"
echo "  Off-target: AUROC >= 0.99"
echo "  Statistical: p < 0.001 (Wilcoxon signed-rank)"
echo "  Conformal: coverage 0.90 +/- 0.02"
echo "  Designer: S = 1.0*mu - 0.5*R - 0.2*sigma"
echo ""

echo "Results from logs:"
echo ""
echo "=== MULTIMODAL V6 ==="
tail -20 logs/multimodal_v6.log 2>/dev/null | grep -E "FINAL|Test Rho|Best Val" || echo "Still in progress or check logs/multimodal_v6.log"
echo ""

echo "=== OFF-TARGET V6 ==="
tail -20 logs/off_target_v6.log 2>/dev/null | grep -E "ENSEMBLE|AUROC|Mean AUROC" || echo "Still in progress or check logs/off_target_v6.log"
echo ""

echo "==========================================================="
echo "Pipeline complete!"
echo "==========================================================="
