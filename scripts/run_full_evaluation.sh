#!/bin/bash
# Master pipeline script for dissertation results evaluation
# PhD: Genome-wide gRNA safety and efficacy modeling

set -e

echo "Starting full evaluation pipeline for ChromaGuide PhD Proposal..."

# 1. Download results from Narval
echo "--------------------------------------------------------"
echo "Step 1: Downloading production weights and results..."
bash scripts/download_results.sh

# 2. Run statistical evaluation scripts
echo "--------------------------------------------------------"
echo "Step 2: Performing primary metric evaluation (AUROC/Spearman)..."
python scripts/evaluate_all_targets.py --results_dir results

# 3. Analyze and visualize results
echo "--------------------------------------------------------"
echo "Step 3: Generating analysis plots and metric reports..."
python scripts/analyze_results.py

# 4. Calibrate Conformal prediction
echo "--------------------------------------------------------"
echo "Step 4: Calculating conformal calibration and prediction sets..."
python scripts/calibrate_conformal.py

# 5. Bootstrap significance testing
echo "--------------------------------------------------------"
echo "Step 5: Running bootstrap significance and Wilcoxon tests..."
python scripts/run_bootstrap_testing.py

# 6. Designer ranking scores
echo "--------------------------------------------------------"
echo "Step 6: Generating integrated designer ranking candidates..."
python scripts/run_designer.py --weights balanced

# 7. Summary and Final report
echo "--------------------------------------------------------"
echo "Step 7: Generating final summary report for Chapter 4/5..."
python scripts/generate_final_report.py

echo "--------------------------------------------------------"
echo "Pipeline complete. Evaluation results stored in results/ directory."
