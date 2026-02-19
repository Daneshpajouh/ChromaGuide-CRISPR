#!/bin/bash
################################################################################
# ChromaGuide V2 - Master Execution Script
#
# Orchestrates all training phases on Narval:
# PHASE 1: DeepHF training (primary dataset)
# PHASE 2: CRISPRon training (cross-dataset validation)
# PHASE 3: Backbone ablation study (architecture comparison)
# PHASE 4: Statistical evaluation (significance testing + SOTA comparison)
#
# Execution: bash scripts/execute_chromaguide_v2.sh
################################################################################

set -e

# Configuration
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
NARVAL_USER="${NARVAL_USER:-daneshpajouh}"
NARVAL_HOST="narval.computecanada.ca"
NARVAL_WORKDIR="~/chromaguide_experiments"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="execution.log"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

################################################################################
# PHASE 0: Pre-Flight Checks
################################################################################

phase_preflight() {
    log_info "════════════════════════════════════════════════════════════════════"
    log_info "PHASE 0: Pre-Flight Checks"
    log_info "════════════════════════════════════════════════════════════════════"

    # Check files exist
    log_info "Verifying SLURM scripts..."

    local scripts=(
        "scripts/slurm_train_v2_deephf.sh"
        "scripts/slurm_train_v2_crispron.sh"
        "scripts/slurm_backbone_ablation.sh"
        "scripts/slurm_statistical_eval.sh"
    )

    for script in "${scripts[@]}"; do
        if [ -f "$PROJECT_ROOT/$script" ]; then
            log_success "  ✓ $script"
        else
            log_error "  ✗ $script NOT FOUND"
            return 1
        fi
    done

    # Check training script
    if [ -f "$PROJECT_ROOT/src/training/train_chromaguide_v2.py" ]; then
        log_success "  ✓ src/training/train_chromaguide_v2.py"
    else
        log_error "  ✗ src/training/train_chromaguide_v2.py NOT FOUND"
        return 1
    fi

    # Check git status
    log_info "Verifying git status..."
    cd "$PROJECT_ROOT"
    if git diff-index --quiet HEAD --; then
        log_success "  ✓ Git working directory clean"
    else
        log_warning "  ⚠ Git working directory has uncommitted changes"
        git status --short | head -5
    fi

    log_success "Pre-flight checks PASSED"
    return 0
}

################################################################################
# PHASE 1: Submit DeepHF Training
################################################################################

phase_submit_deephf() {
    log_info "════════════════════════════════════════════════════════════════════"
    log_info "PHASE 1: Submitting DeepHF Training Job"
    log_info "════════════════════════════════════════════════════════════════════"

    log_info "Job Details:"
    log_info "  Dataset: DeepHF (~40K sgRNAs)"
    log_info "  GPUs: 4 × H100"
    log_info "  Walltime: 12 hours"
    log_info "  Split: Gene-held-out (most stringent)"
    log_info "  Target: Spearman ρ >= 0.911 (beat CCL/MoFF baseline)"

    local script="$PROJECT_ROOT/scripts/slurm_train_v2_deephf.sh"

    log_info "Submitting to Narval..."
    read -p "Ready to submit DeepHF job? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Make script executable
        chmod +x "$script"

        # Submit job - LOCAL execution (for testing) or NARVAL
        if [ "$1" == "local" ]; then
            log_warning "Local execution mode (testing)"
            bash "$script" &
            local job_pid=$!
            log_success "Started locally with PID: $job_pid"
            echo "$job_pid" > ".deephf_pid"
        else
            # Submit to Narval via SSH
            log_info "Submitting via SSH to narval.computecanada.ca..."
            ssh -t ${NARVAL_USER}@${NARVAL_HOST} \
                "cd ${NARVAL_WORKDIR} && sbatch $(basename $script)" \
                2>&1 | tee -a "$LOG_FILE"

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                log_success "DeepHF job submitted successfully"
            else
                log_error "Failed to submit DeepHF job"
                return 1
            fi
        fi
    else
        log_warning "Skipped DeepHF submission"
    fi

    return 0
}

################################################################################
# PHASE 2: Submit CRISPRon Training (optional, parallel)
################################################################################

phase_submit_crispron() {
    log_info "════════════════════════════════════════════════════════════════════"
    log_info "PHASE 2: Submitting CRISPRon Training Job (Optional)"
    log_info "════════════════════════════════════════════════════════════════════"

    log_info "Job Details:"
    log_info "  Dataset: CRISPRon (cross-dataset validation)"
    log_info "  GPUs: 4 × H100"
    log_info "  Walltime: 12 hours"
    log_info "  Split: Dataset-held-out"
    log_info "  Target: Spearman ρ >= 0.876 (beat ChromeCRISPR baseline)"

    log_info ""
    read -p "Submit CRISPRon job in parallel? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        local script="$PROJECT_ROOT/scripts/slurm_train_v2_crispron.sh"
        chmod +x "$script"

        if [ "$1" == "local" ]; then
            log_warning "Local execution mode (testing)"
            bash "$script" &
            local job_pid=$!
            log_success "Started locally with PID: $job_pid"
            echo "$job_pid" > ".crispron_pid"
        else
            log_info "Submitting via SSH..."
            ssh -t ${NARVAL_USER}@${NARVAL_HOST} \
                "cd ${NARVAL_WORKDIR} && sbatch $(basename $script)" \
                2>&1 | tee -a "$LOG_FILE"

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                log_success "CRISPRon job submitted successfully"
            else
                log_error "Failed to submit CRISPRon job"
                return 1
            fi
        fi
    else
        log_warning "Skipped CRISPRon submission"
    fi

    return 0
}

################################################################################
# PHASE 3: Monitor Training Progress
################################################################################

phase_monitor() {
    log_info "════════════════════════════════════════════════════════════════════"
    log_info "PHASE 3: Monitoring Training Progress"
    log_info "════════════════════════════════════════════════════════════════════"

    log_info "Monitoring job status on Narval..."
    log_info "Commands to check progress:"
    log_info ""
    log_info "  ssh ${NARVAL_USER}@${NARVAL_HOST}"
    log_info "  cd ${NARVAL_WORKDIR}"
    log_info "  squeue -u $NARVAL_USER  # View active jobs"
    log_info "  tail -f logs/train_deephf_*.out  # View logs"
    log_info "  sacct -u $NARVAL_USER  # View completed jobs"
    log_info ""

    local check_interval=300  # 5 minutes
    local max_wait=86400     # 24 hours
    local elapsed=0

    if [ "$1" != "local" ]; then
        log_info "Waiting for jobs to complete (max ${max_wait}s)..."

        while [ $elapsed -lt $max_wait ]; do
            # Check if jobs are still queued/running
            ssh ${NARVAL_USER}@${NARVAL_HOST} \
                "squeue -u $NARVAL_USER | grep chromaguide" \
                > /dev/null 2>&1

            if [ $? -ne 0 ]; then
                log_success "All jobs completed!"
                break
            fi

            log_info "Jobs still running... (elapsed: ${elapsed}s)"
            sleep $check_interval
            elapsed=$((elapsed + check_interval))
        done
    fi

    return 0
}

################################################################################
# PHASE 4: Post-Training Evaluation
################################################################################

phase_postprocessing() {
    log_info "════════════════════════════════════════════════════════════════════"
    log_info "PHASE 4: Submitting Post-Training Evaluation"
    log_info "════════════════════════════════════════════════════════════════════"

    log_info "Waiting for training complete, then submitting evaluation..."

    read -p "Submit statistical evaluation job? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        local script="$PROJECT_ROOT/scripts/slurm_statistical_eval.sh"
        chmod +x "$script"

        if [ "$1" == "local" ]; then
            log_warning "Local execution mode (testing)"
            bash "$script" &
            local job_pid=$!
            log_success "Started locally with PID: $job_pid"
        else
            log_info "Submitting via SSH..."
            ssh -t ${NARVAL_USER}@${NARVAL_HOST} \
                "cd ${NARVAL_WORKDIR} && sbatch scripts/slurm_statistical_eval.sh" \
                2>&1 | tee -a "$LOG_FILE"

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                log_success "Statistical evaluation job submitted"
            else
                log_error "Failed to submit evaluation job"
                return 1
            fi
        fi
    else
        log_warning "Skipped evaluation submission"
    fi

    return 0
}

################################################################################
# PHASE 5: Ablation Study (Optional)
################################################################################

phase_ablation() {
    log_info "════════════════════════════════════════════════════════════════════"
    log_info "PHASE 5: Backbone Ablation Study (Optional)"
    log_info "════════════════════════════════════════════════════════════════════"

    log_info "Job Details:"
    log_info "  Tests: 5 DNA encoding architectures"
    log_info "  GPUs: 4 × H100"
    log_info "  Walltime: 24 hours"
    log_info "  Purpose: Architecture comparison with fixed training budget"

    read -p "Submit backbone ablation study? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        local script="$PROJECT_ROOT/scripts/slurm_backbone_ablation.sh"
        chmod +x "$script"

        if [ "$1" == "local" ]; then
            log_warning "Local execution mode (testing)"
            bash "$script" &
            local job_pid=$!
            log_success "Started locally with PID: $job_pid"
        else
            log_info "Submitting via SSH..."
            ssh -t ${NARVAL_USER}@${NARVAL_HOST} \
                "cd ${NARVAL_WORKDIR} && sbatch $(basename $script)" \
                2>&1 | tee -a "$LOG_FILE"

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                log_success "Ablation study job submitted"
            else
                log_error "Failed to submit ablation job"
                return 1
            fi
        fi
    else
        log_warning "Skipped ablation submission"
    fi

    return 0
}

################################################################################
# Main Execution
################################################################################

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                        ║"
    echo "║                  ChromaGuide V2 - Master Execution Script              ║"
    echo "║                                                                        ║"
    echo "║              Orchestrates Complete Training Pipeline on Narval        ║"
    echo "║                                                                        ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""

    cd "$PROJECT_ROOT"

    # Parse arguments
    local exec_mode="narval"  # Default: submit to Narval
    if [ "$1" == "local" ]; then
        exec_mode="local"
        log_warning "Running in LOCAL testing mode (not submitting to Narval)"
    fi

    # Run phases
    phase_preflight || { log_error "Pre-flight checks failed"; exit 1; }
    phase_submit_deephf "$exec_mode" || { log_error "DeepHF submission failed"; exit 1; }
    phase_submit_crispron "$exec_mode" || { log_error "CRISPRon submission failed"; exit 1; }
    phase_monitor "$exec_mode"
    phase_postprocessing "$exec_mode"
    phase_ablation "$exec_mode"

    # Final summary
    log_success "════════════════════════════════════════════════════════════════════"
    log_success "EXECUTION COMPLETE"
    log_success "════════════════════════════════════════════════════════════════════"

    echo ""
    log_info "Summary:"
    log_info "  1. All training jobs submitted to Narval"
    log_info "  2. Logs: ~/chromaguide_experiments/logs/"
    log_info "  3. Results: ~/chromaguide_experiments/checkpoints/ + results/"
    log_info "  4. GitHub: Auto-commit after each phase"
    log_info ""
    log_info "Next steps:"
    log_info "  • Monitor: ssh ${NARVAL_USER}@${NARVAL_HOST} && squeue -u $NARVAL_USER"
    log_info "  • Check logs: tail -f ~/chromaguide_experiments/logs/train_deephf_*.out"
    log_info "  • Pull results: git pull origin main (gets auto-committed)"
    log_info ""
}

# Run main with arguments
main "$@"
