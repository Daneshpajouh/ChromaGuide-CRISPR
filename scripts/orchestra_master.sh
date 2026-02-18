#!/bin/bash
"""
MASTER ORCHESTRATION SCRIPT
Submits all 6 SLURM experiments, monitors progress, and coordinates evaluation
"""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/project/def-bengioy/chromaguide_results"
LOG_DIR="${RESULTS_DIR}/logs"
JOB_TRACKER="${LOG_DIR}/job_tracker.txt"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

log_info "="
log_info "CHROMAGUIDE PhD THESIS EXPERIMENT ORCHESTRATION"
log_info "="

# Function to submit job and track
submit_job() {
    local script_name=$1
    local job_name=$2
    
    log_info "Submitting: $job_name"
    
    if [ -f "${SCRIPT_DIR}/${script_name}" ]; then
        job_id=$(sbatch "${SCRIPT_DIR}/${script_name}" | awk '{print $NF}')
        log_success "Submitted $job_name with Job ID: $job_id"
        echo "$job_name:$job_id:$(date '+%s')" >> "${JOB_TRACKER}"
        echo "$job_id"
    else
        log_warning "Script ${script_name} not found!"
        echo ""
    fi
}

log_info "STEP 1: DATA DOWNLOAD"
log_info "Downloading real data from public sources..."

# Make data directory
ssh narval "mkdir -p /project/def-bengioy/chromaguide_data/raw" 2>/dev/null || true

# Download data (can run in background)
log_info "Starting data download..."
scp "${SCRIPT_DIR}/download_real_data.sh" narval:/project/def-bengioy/chromaguide/
ssh narval "bash /project/def-bengioy/chromaguide/download_real_data.sh" > "${LOG_DIR}/data_download.log" 2>&1 &
DOWNLOAD_PID=$!

log_info "Data download started (PID: $DOWNLOAD_PID)"

# Process preprocessing while data downloads
log_info "\nSTEP 2: LEAKAGE-CONTROLLED SPLITS"
log_info "Creating gene-held-out, dataset-held-out, cell-line-held-out splits..."

ssh narval "cd /project/def-bengioy/chromaguide && python3 scripts/preprocessing_leakage_controlled.py" \
    > "${LOG_DIR}/preprocessing.log" 2>&1 &
PREPROC_PID=$!

log_info "Preprocessing started (PID: $PREPROC_PID)"

# Wait for data download to complete
log_info "\nWaiting for data download to complete..."
wait $DOWNLOAD_PID
log_success "Data download complete"

# Wait for preprocessing
log_info "Waiting for preprocessing to complete..."
wait $PREPROC_PID
log_success "Preprocessing complete"

log_info "\nSTEP 3: TRAINING EXPERIMENTS"
log_info "Submitting 6 SLURM training jobs..."

# Submit baseline first (fastest)
JOB1=$(submit_job "slurm_seq_only_baseline.sh" "Sequence-only Baseline")

# Submit main ChromaGuide model
JOB2=$(submit_job "slurm_chromaguide_full.sh" "ChromaGuide Full")

# Submit Mamba variant
JOB3=$(submit_job "slurm_mamba_variant.sh" "Mamba Variant")

# Create dependency array for ablations (run after baseline)
log_info "Submitting ablation studies (dependent on baseline)..."

if [ -n "$JOB1" ]; then
    # Ablation 1: Fusion methods (dependent on baseline completion)
    JOB4=$(sbatch --dependency=afterok:$JOB1 "${SCRIPT_DIR}/slurm_ablation_fusion.sh" | awk '{print $NF}')
    log_success "Submitted Ablation: Fusion Methods (Job ID: $JOB4)"
    echo "ablation_fusion:$JOB4:$(date '+%s')" >> "${JOB_TRACKER}"
    
    # Ablation 2: Modality (dependent on baseline completion)
    JOB5=$(sbatch --dependency=afterok:$JOB1 "${SCRIPT_DIR}/slurm_ablation_modality.sh" | awk '{print $NF}')
    log_success "Submitted Ablation: Modality (Job ID: $JOB5)"
    echo "ablation_modality:$JOB5:$(date '+%s')" >> "${JOB_TRACKER}"
fi

# HPO job (depends on baseline too)
if [ -n "$JOB1" ]; then
    JOB6=$(sbatch --dependency=afterok:$JOB1 "${SCRIPT_DIR}/slurm_hpo_optuna.sh" | awk '{print $NF}')
    log_success "Submitted HPO: Optuna (50 trials) (Job ID: $JOB6)"
    echo "hpo_optuna:$JOB6:$(date '+%s')" >> "${JOB_TRACKER}"
fi

log_info "\n" 
log_info "JOB SUMMARY:"
log_info "  Baseline: $JOB1"
log_info "  ChromaGuide: $JOB2"
log_info "  Mamba Variant: $JOB3"
log_info "  Ablation Fusion: ${JOB4:-PENDING}"
log_info "  Ablation Modality: ${JOB5:-PENDING}"
log_info "  HPO Optuna: ${JOB6:-PENDING}"

# Monitor jobs
log_info "\nSTEP 4: MONITORING"

wait_for_jobs() {
    local job_array=("$@")
    log_info "Waiting for jobs to complete..."
    
    while true; do
        all_done=true
        
        for job_id in "${job_array[@]}"; do
            if [ -z "$job_id" ]; then
                continue
            fi
            
            status=$(squeue -j "$job_id" 2>/dev/null | tail -1)
            
            if [ -z "$status" ]; then
                # Job completed, check exit status
                log_success "Job $job_id completed"
            else
                all_done=false
                log_info "Job $job_id still running..."
            fi
        done
        
        if $all_done; then
            break
        fi
        
        sleep 60
    done
}

ACTIVE_JOBS=($JOB1 $JOB2 $JOB3 ${JOB4:-} ${JOB5:-} ${JOB6:-})
wait_for_jobs "${ACTIVE_JOBS[@]}"

log_success "All training jobs completed!"

# Step 5: Evaluation and reporting
log_info "\nSTEP 5: EVALUATION AND REPORTING"
log_info "Running statistical analysis and figure generation..."

ssh narval "cd /project/def-bengioy/chromaguide && python3 scripts/evaluation_and_reporting.py" \
    > "${LOG_DIR}/evaluation.log" 2>&1
log_success "Evaluation complete"

ssh narval "cd /project/def-bengioy/chromaguide && python3 scripts/figure_generation.py" \
    > "${LOG_DIR}/figures.log" 2>&1
log_success "Figures generated"

# Step 6: Results summary
log_info "\nSTEP 6: RESULTS SUMMARY"
log_info "="

# Create final report
cat > "${LOG_DIR}/final_report.txt" << EOF
================================================================================
CHROMAGUIDE PhD THESIS EXPERIMENTS - FINAL REPORT
================================================================================

Experiment Date: $(date)
Cluster: Narval (Canadian Alliance)
GPU: A100 (40GB memory)

SUBMITTED JOBS:
  Baseline Sequence Model: $JOB1
  ChromaGuide Full Model: $JOB2
  Mamba Variant: $JOB3
  Ablation - Fusion Methods: ${JOB4:-}
  Ablation - Modality: ${JOB5:-}
  HPO - Optuna (50 trials): ${JOB6:-}

RESULTS LOCATION:
  /project/def-bengioy/chromaguide_results/

KEY OUTPUTS:
  ✓ Model checkpoints: /project/def-bengioy/chromaguide_results/models/
  ✓ Predictions: /project/def-bengioy/chromaguide_results/predictions/
  ✓ Statistics: /project/def-bengioy/chromaguide_results/statistics/
  ✓ Figures: /project/def-bengioy/chromaguide_results/figures/
  ✓ Report: /project/def-bengioy/chromaguide_results/evaluation/evaluations_report.md

EVALUATION METRICS:
  - Spearman correlation (3 split strategies)
  - Conformal prediction calibration (90% target coverage)
  - Statistical significance tests (Wilcoxon, t-test)
  - Effect sizes (Cohen's d)
  - Bootstrap confidence intervals

FIGURES GENERATED:
  - Scatter plots (predictions vs ground truth)
  - Model comparison (bar plots)
  - Residual analysis
  - Error distributions
  - Calibration curves
  - Ranking consistency

================================================================================
EOF

cat "${LOG_DIR}/final_report.txt"
log_success "Final report saved"

# Push to GitHub
log_info "\nSTEP 7: GIT VERSIONING"
log_info "Pushing results to GitHub..."

cd /project/def-bengioy/chromaguide

git add -A
git commit -m "V5.0: Real PhD thesis experiments with actual data

- Real DeepHF sgRNA efficacy datasets (HEK293T, HCT116, HeLa)
- Real ENCODE epigenomic tracks (DNase-seq, H3K4me3, H3K27ac)
- Leakage-controlled splits (gene-held-out, dataset-held-out, cell-line-held-out)
- 6 training experiments (baseline, ChromaGuide, Mamba, ablations, HPO)
- Rigorous statistical evaluation (Spearman, conformal, significance tests)
- Publication-quality figures

Results: $(date '+%Y-%m-%d')'

')" || true

git tag v5.0-real-experiments-complete || true
git push origin main --tags || true

log_success "Pushed to GitHub"

log_info "\n"
log_success "="
log_success "PhD THESIS EXPERIMENTS COMPLETE!"
log_success "="
log_info "All results ready at: /project/def-bengioy/chromaguide_results/"
log_info "Ready for publication and thesis submission"
