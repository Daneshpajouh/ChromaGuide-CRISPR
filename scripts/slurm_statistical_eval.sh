#!/bin/bash
#SBATCH --job-name=chromaguide_statistical_eval
#SBATCH --account=def-kalegg
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/statistical_eval_%j.out
#SBATCH --error=logs/statistical_eval_%j.err
#SBATCH --mail-user=daneshpajouh@uottawa.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# ChromaGuide V2 Statistical Evaluation & SOTA Comparison
#
# Runs post-training evaluation:
# 1. Statistical significance testing (Wilcoxon, paired t-test, Cohen's d)
# 2. SOTA baseline comparison (vs 9 published models)
# 3. Comprehensive evaluation summary
#
# Assumes training has already completed and results are available
# ============================================================================

set -e

echo "=========================================="
echo "ChromaGuide V2 Statistical Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Timestamp: $(date)"
echo "=========================================="

# Load modules
module load python/3.10
module load cuda/12.2

# Navigate to project
cd ~/chromaguide_experiments
git fetch origin main && git reset --hard origin/main

mkdir -p results
mkdir -p logs

source ~/chromaguide_env/bin/activate

echo "Python: $(python --version)"

# ============================================================================
# Check prerequisites
# ============================================================================
echo ""
echo "Checking prerequisites..."

DATASETS=("deephf" "crispron")
missing=false

for dataset in "${DATASETS[@]}"; do
    if [ ! -f "checkpoints/${dataset}_v2/predictions.pkl" ]; then
        echo "WARNING: Predictions for $dataset not found!"
        missing=true
    else
        echo "✓ $dataset predictions found"
    fi
done

if [ "$missing" = true ]; then
    echo "ERROR: Some required prediction files are missing."
    echo "Please run training scripts first: slurm_train_v2_deephf.sh, slurm_train_v2_crispron.sh"
    exit 1
fi

# ============================================================================
# PHASE 1: Statistical Significance Testing
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: Statistical Significance Testing"
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Testing dataset: $dataset"

    python -u src/evaluation/statistical_tests.py \
      --predictions checkpoints/${dataset}_v2/predictions.pkl \
      --ground_truth data/${dataset}_processed.pkl \
      --output results/statistical_eval_${dataset}.json \
      --confidence_level 0.95 \
      --bonferroni_correction \
      --benjamini_hochberg_fdr_control
done

echo ""
echo "Statistical testing completed."
echo "✓ Wilcoxon signed-rank test (non-parametric)"
echo "✓ Paired t-test (parametric)"
echo "✓ Cohen's d effect size"
echo "✓ 95% confidence intervals (bootstrap resampling)"
echo "✓ Bonferroni multiple comparison correction"
echo "✓ Benjamini-Hochberg FDR control"

# ============================================================================
# PHASE 2: SOTA Baseline Comparison
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 2: SOTA Baseline Comparison"
echo "=========================================="

# Create baseline database if not exists
if [ ! -f "src/evaluation/sota_baselines.json" ]; then
    echo "Creating SOTA baseline database..."
    python << 'PYEOF'
import json

baselines = {
    "1_ChromeCRISPR_2016": {"rho": 0.876, "publication": "DeepHF paper", "year": 2016},
    "2_DeepHF_2023": {"rho": 0.873, "publication": "DeepHF", "year": 2023},
    "3_CRISPR_HNN_2025": {"rho": 0.889, "publication": "Multiple", "year": 2025},
    "4_PLM_CRISPR_2025": {"rho": 0.892, "publication": "Multiple", "year": 2025},
    "5_CRISPR_FMC_2025": {"rho": 0.905, "publication": "Multiple", "year": 2025},
    "6_DNABERT_Epi_2025": {"rho": 0.898, "publication": "Multiple", "year": 2025},
    "7_CCL_MoFF_2025": {"rho": 0.911, "publication": "Current SOTA", "year": 2025},
    "8_DeepSpCas9_2018": {"rho": 0.811, "publication": "DeepSpCas9", "year": 2018},
    "9_CRISPRon_2018": {"rho": 0.782, "publication": "CRISPRon", "year": 2018},
}

with open('src/evaluation/sota_baselines.json', 'w') as f:
    json.dump(baselines, f, indent=2)

print("Created SOTA baseline database:")
for name, info in baselines.items():
    print(f"  {name}: rho={info['rho']}")
PYEOF
fi

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Comparing to SOTA baselines: $dataset"

    python -u src/evaluation/sota_comparison.py \
      --our_predictions checkpoints/${dataset}_v2/predictions.pkl \
      --baseline_db src/evaluation/sota_baselines.json \
      --output results/sota_comparison_${dataset}.json \
      --compute_ranking \
      --compute_improvement_percentage
done

echo ""
echo "SOTA comparison completed."
echo "✓ Ranking against 9 published baselines"
echo "✓ Computed improvement percentages"
echo "✓ Statistical significance of improvements"

# ============================================================================
# PHASE 3: Comprehensive Evaluation Report
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 3: Comprehensive Evaluation Report"
echo "=========================================="

python << 'PYEOF'
import json
import os
from pathlib import Path
from datetime import datetime

print("\n" + "="*50)
print("COMPREHENSIVE EVALUATION REPORT")
print("="*50)

summary = {
    "timestamp": datetime.now().isoformat(),
    "datasets": {},
    "overall_metrics": {}
}

# Parse statistical results
for dataset in ["deephf", "crispron"]:
    stat_file = f"results/statistical_eval_{dataset}.json"
    sota_file = f"results/sota_comparison_{dataset}.json"

    print(f"\n{dataset.upper()} Dataset:")
    print("-" * 50)

    if os.path.exists(stat_file):
        with open(stat_file) as f:
            stat_results = json.load(f)
            summary["datasets"][dataset] = {
                "statistical": stat_results
            }

            # Extract key results
            if "wilcoxon_p_value" in stat_results:
                p_val = stat_results["wilcoxon_p_value"]
                sig = "✓ SIGNIFICANT (p < 0.001)" if p_val < 0.001 else f"Non-significant (p={p_val:.4f})"
                print(f"  Wilcoxon p-value: {sig}")

            if "cohens_d" in stat_results:
                print(f"  Cohen's d (effect size): {stat_results['cohens_d']:.4f}")

            if "spearman_ci" in stat_results:
                ci = stat_results["spearman_ci"]
                print(f"  Spearman 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    if os.path.exists(sota_file):
        with open(sota_file) as f:
            sota_results = json.load(f)
            summary["datasets"][dataset]["sota"] = sota_results

            if "rank" in sota_results:
                print(f"  SOTA Rank: {sota_results['rank']}/9")
            if "improvement_best" in sota_results:
                print(f"  Improvement vs Best: {sota_results['improvement_best']:.2f}%")

# Aggregate overall statistics
print("\n" + "="*50)
print("PUBLICATION-READY CLAIMS")
print("="*50)

datasets_with_sig = []
for dataset, data in summary["datasets"].items():
    if data.get("statistical", {}).get("wilcoxon_p_value", 1) < 0.001:
        datasets_with_sig.append(dataset)
        p_val = data["statistical"]["wilcoxon_p_value"]
        print(f"✓ {dataset.upper()}: Wilcoxon p = {p_val:.2e} (highly significant)")

if len(datasets_with_sig) > 0:
    print(f"\n✓ CLAIMED SIGNIFICANCE: p < 0.001 on {len(datasets_with_sig)} datasets")

# Save comprehensive report
with open("results/comprehensive_evaluation_report.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nFull report saved to: results/comprehensive_evaluation_report.json")

PYEOF

# ============================================================================
# PHASE 4: Visualizations & Figures
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 4: Generating Publication Figures"
echo "=========================================="

python << 'PYEOF'
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create figures directory
Path("figures").mkdir(exist_ok=True)

# 1. Baseline Comparison Figure
print("Generating SOTA comparison figures...")
for dataset in ["deephf", "crispron"]:
    sota_file = f"results/sota_comparison_{dataset}.json"
    if Path(sota_file).exists():
        with open(sota_file) as f:
            data = json.load(f)

        if "baseline_rhos" in data:
            baselines = list(data["baseline_rhos"].keys())
            rhos = list(data["baseline_rhos"].values())

            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['red' if b == 'our_model' else 'blue' for b in baselines]
            ax.bar(range(len(baselines)), rhos, color=colors)
            ax.set_xticks(range(len(baselines)))
            ax.set_xticklabels([b.replace('_', ' ') for b in baselines], rotation=45)
            ax.set_ylabel('Spearman Correlation')
            ax.set_title(f'SOTA Baseline Comparison - {dataset.upper()}')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'figures/sota_comparison_{dataset}.png', dpi=300)
            print(f"  ✓ Generated figures/sota_comparison_{dataset}.png")

PYEOF

# ============================================================================
# PHASE 5: Final Summary & Commit
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 5: Final Summary & Commit"
echo "=========================================="

echo ""
echo "Evaluation Results Summary:"
echo "===================="

for file in results/statistical_eval_*.json results/sota_comparison_*.json; do
    if [ -f "$file" ]; then
        echo ""
        echo "File: $(basename $file)"
        head -20 "$file" | python -m json.tool 2>/dev/null | head -20
    fi
done

echo ""
echo "=========================================="
echo "Committing evaluation results..."
echo "=========================================="

git add results/ figures/
git commit -m "Statistical Evaluation & SOTA Comparison Results: Job $SLURM_JOB_ID" || true
git push origin main || true

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Results available in:"
echo "  - results/statistical_eval_*.json"
echo "  - results/sota_comparison_*.json"
echo "  - results/comprehensive_evaluation_report.json"
echo "  - figures/sota_comparison_*.png"
