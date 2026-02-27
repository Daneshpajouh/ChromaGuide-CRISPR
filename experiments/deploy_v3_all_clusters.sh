#!/bin/bash
# ChromaGuide v3 Deployment Script
# Deploys code, data, and submits jobs to all 4 clusters

set -e

CLUSTERS=("nibi" "narval" "rorqual" "fir")
SCRATCH_DIR="~/scratch/chromaguide_v3"
VENV_DIR="~/scratch/chromaguide_v2_env"
LOCAL_REPO="/home/user/workspace/chromaguide-repo"

# Files to deploy
V3_FILES=(
    "chromaguide/modules/sequence_encoders_v3.py"
    "chromaguide/modules/conformal_v3.py"
    "chromaguide/training/losses_v3.py"
    "experiments/train_experiment_v3.py"
    "experiments/generate_slurm_v3.py"
)

# Existing files needed
EXISTING_FILES=(
    "chromaguide/__init__.py"
    "chromaguide/configs/__init__.py"
    "chromaguide/configs/default.yaml"
    "chromaguide/configs/caduceus.yaml"
    "chromaguide/configs/dnabert2.yaml"
    "chromaguide/configs/evo.yaml"
    "chromaguide/configs/nucleotide_transformer.yaml"
    "chromaguide/data/__init__.py"
    "chromaguide/data/acquire.py"
    "chromaguide/data/dataset.py"
    "chromaguide/data/preprocess.py"
    "chromaguide/data/splits.py"
    "chromaguide/evaluation/__init__.py"
    "chromaguide/evaluation/metrics.py"
    "chromaguide/models/__init__.py"
    "chromaguide/models/chromaguide.py"
    "chromaguide/modules/__init__.py"
    "chromaguide/modules/conformal.py"
    "chromaguide/modules/epigenomic_encoder.py"
    "chromaguide/modules/fusion.py"
    "chromaguide/modules/prediction_head.py"
    "chromaguide/modules/sequence_encoders.py"
    "chromaguide/training/__init__.py"
    "chromaguide/training/losses.py"
    "chromaguide/training/trainer.py"
    "chromaguide/utils/__init__.py"
    "chromaguide/utils/config.py"
    "chromaguide/utils/reproducibility.py"
    "experiments/prepare_data.py"
)

echo "============================================"
echo "ChromaGuide v3 Deployment"
echo "Date: $(date)"
echo "============================================"

for cluster in "${CLUSTERS[@]}"; do
    echo ""
    echo "--- Deploying to ${cluster} ---"
    
    HOST="${cluster}"
    
    # Create v3 directory structure
    echo "  Creating directory structure..."
    ssh ${HOST} "
        mkdir -p ${SCRATCH_DIR}/{chromaguide/{configs,data,evaluation,models,modules,training,utils},experiments,data,results_v3,logs,model_cache}
    " 2>/dev/null || { echo "  FAILED to connect to ${cluster}"; continue; }
    
    # Deploy new v3 files
    echo "  Deploying v3 code..."
    for f in "${V3_FILES[@]}"; do
        local_path="${LOCAL_REPO}/${f}"
        if [ -f "${local_path}" ]; then
            scp -q "${local_path}" "${HOST}:${SCRATCH_DIR}/${f}" 2>/dev/null
        else
            echo "  WARNING: ${f} not found locally"
        fi
    done
    
    # Deploy existing files (needed for imports)
    echo "  Deploying existing modules..."
    for f in "${EXISTING_FILES[@]}"; do
        local_path="${LOCAL_REPO}/${f}"
        if [ -f "${local_path}" ]; then
            scp -q "${local_path}" "${HOST}:${SCRATCH_DIR}/${f}" 2>/dev/null
        fi
    done
    
    # Deploy SLURM jobs
    echo "  Deploying SLURM scripts..."
    scp -q ${LOCAL_REPO}/experiments/slurm_v3_${cluster}/*.sh "${HOST}:${SCRATCH_DIR}/experiments/" 2>/dev/null
    
    # Create symlink for data if v2 data exists
    echo "  Setting up data..."
    ssh ${HOST} "
        if [ -d ~/scratch/chromaguide_v2/data ] && [ ! -d ${SCRATCH_DIR}/data/processed ]; then
            ln -sf ~/scratch/chromaguide_v2/data/processed ${SCRATCH_DIR}/data/processed
            ln -sf ~/scratch/chromaguide_v2/data/raw ${SCRATCH_DIR}/data/raw
            echo '    Linked data from v2'
        elif [ -d ${SCRATCH_DIR}/data/processed ]; then
            echo '    Data already exists'
        else
            echo '    WARNING: No data found! Run prepare_data.py first'
        fi
    " 2>/dev/null
    
    # Submit jobs
    echo "  Submitting v3 jobs..."
    ssh ${HOST} "
        cd ${SCRATCH_DIR}/experiments
        submitted=0
        for script in cg3_*.sh; do
            if [ -f \"\${script}\" ]; then
                jobid=\$(sbatch \${script} 2>&1 | grep -oP '\\d+' || echo 'FAILED')
                if [ \"\${jobid}\" != 'FAILED' ]; then
                    submitted=\$((submitted + 1))
                fi
                sleep 0.5
            fi
        done
        echo \"    Submitted \${submitted} jobs on ${cluster}\"
        echo \"    Job queue:\"
        squeue -u \$(whoami) --format='%.10i %.20j %.8T %.10M %.5D %R' | head -5
        echo '    ...'
    " 2>/dev/null
    
    echo "  ${cluster} deployment complete"
done

echo ""
echo "============================================"
echo "V3 Deployment Complete"
echo "============================================"
