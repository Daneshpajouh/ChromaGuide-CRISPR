#!/bin/bash
# Master pipeline script for dissertation results evaluation
# PhD: Genome-wide gRNA safety and efficacy modeling

set -e

echo "Starting full evaluation pipeline for ChromaGuide PhD Proposal..."

# 1. Optional sync step
echo "--------------------------------------------------------"
echo "Step 1: Syncing production weights/results (optional)..."
if [ -f scripts/download_results.sh ]; then
  bash scripts/download_results.sh
else
  echo "download_results.sh not found; continuing with local results/"
fi

# 2. Run statistical evaluation scripts
echo "--------------------------------------------------------"
echo "Step 2: Performing primary metric evaluation (AUROC/Spearman)..."
python scripts/evaluate_all_targets.py --results_dir results

# 3. Analyze and visualize results (optional)
echo "--------------------------------------------------------"
echo "Step 3: Generating analysis plots and metric reports..."
if [ -f scripts/analyze_results.py ]; then
  python scripts/analyze_results.py
else
  echo "analyze_results.py not found; skipping optional analysis step"
fi

# 4. Calibrate Conformal prediction
echo "--------------------------------------------------------"
echo "Step 4: Calculating conformal calibration and prediction sets..."
python scripts/calibrate_conformal.py

# 5. Bootstrap significance testing
echo "--------------------------------------------------------"
echo "Step 5: Running bootstrap significance and Wilcoxon tests..."
if [ -f results/gold_results_on_target.csv ]; then
  python scripts/run_bootstrap_testing.py --results_a results/gold_results_on_target.csv
else
  echo "results/gold_results_on_target.csv not found; skipping bootstrap step"
fi

# 6. Designer ranking scores (optional, requires model checkpoints + candidates)
echo "--------------------------------------------------------"
echo "Step 6: Generating integrated designer ranking candidates..."
if [ -f models/on_target.pt ] && [ -f data/candidates.csv ]; then
  python scripts/run_designer.py \
    --on-target-model models/on_target.pt \
    --candidates data/candidates.csv \
    --output-dir results/designer_evaluation
else
  echo "Designer inputs missing (models/on_target.pt, data/candidates.csv); skipping"
fi

# 7. Summary and Final report
echo "--------------------------------------------------------"
echo "Step 7: Generating final summary report for Chapter 4/5..."
python scripts/generate_final_report.py

echo "--------------------------------------------------------"
echo "Pipeline complete. Evaluation results stored in results/ directory."
