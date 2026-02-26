#!/usr/bin/env python3
"""Submit all 45 ChromaGuide experiments to Rorqual (H100-80GB).

Rorqual has H100 GPUs — can use larger batch sizes.
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT = os.path.expanduser("~/scratch/chromaguide_v2")
OUTPUT_DIR = os.path.join(PROJECT, "results")
LOGS_DIR = os.path.join(PROJECT, "logs")
SLURM_SCRIPTS_DIR = os.path.join(PROJECT, "slurm_scripts")

CLUSTER = "rorqual"
GPU_TYPE = "1"  # Rorqual uses generic gpu:1
ACCOUNT = "def-kwiese_gpu"

# H100-80GB allows larger batch sizes
BACKBONE_CONFIG = {
    "cnn_gru": {
        "time": "4:00:00",
        "mem": "48G",
        "batch_size": 512,
        "patience": 15,
        "extra_args": "",
    },
    "dnabert2": {
        "time": "8:00:00",
        "mem": "64G",
        "batch_size": 128,
        "patience": 10,
        "extra_args": "",
    },
    "caduceus": {
        "time": "6:00:00",
        "mem": "48G",
        "batch_size": 512,
        "patience": 15,
        "extra_args": "",
    },
    "evo": {
        "time": "10:00:00",
        "mem": "64G",
        "batch_size": 64,
        "patience": 10,
        "extra_args": "",
    },
    "nucleotide_transformer": {
        "time": "14:00:00",
        "mem": "80G",
        "batch_size": 32,
        "patience": 8,
        "extra_args": "",
    },
}

SPLITS = ["A", "B", "C"]
SEEDS = [42, 123, 456]

MODULE_LOADS = """module purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.2
module load arrow/17.0.0
"""

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=cg-{backbone}-{split}-s{seed}
#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:{gpu_type}
#SBATCH --cpus-per-task=8
#SBATCH --output={logs_dir}/cg-{backbone}-{split}-s{seed}-%j.out
#SBATCH --error={logs_dir}/cg-{backbone}-{split}-s{seed}-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amir@mystorax.com

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Backbone: {backbone} | Split: {split} | Seed: {seed}"
echo "Cluster: rorqual (H100-80GB)"
echo "==================="

{module_loads}

source ~/scratch/chromaguide_v2_env/bin/activate

export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=~/scratch/.cache/huggingface
export HF_HOME=~/scratch/.cache/huggingface
export TORCH_HOME=~/scratch/.cache/torch
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd {project}

# Verify real data
NSAMPLES=$(python -c "import numpy as np; print(len(np.load('data/processed/efficacy.npy')))")
echo "Data samples: $NSAMPLES"
if [ "$NSAMPLES" -lt 100000 ]; then
    echo "ERROR: Expected 291K+ samples, got $NSAMPLES"
    exit 1
fi

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "VRAM: $(python -c 'import torch; print(f\"{{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}} GB\")')"
fi

mkdir -p results/{backbone}_split{split}_seed{seed}

python experiments/train_experiment.py \\
    --backbone {backbone} \\
    --split {split} \\
    --split-fold {split_fold} \\
    --seed {seed} \\
    --data-dir {project}/data \\
    --output-dir {project}/results \\
    --no-wandb \\
    --patience {patience} \\
    --gradient-clip 1.0 \\
    --batch-size {batch_size} \\
    {extra_args}

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

if [ -f results/{backbone}_split{split}_seed{seed}/results.json ]; then
    echo "Results:"
    cat results/{backbone}_split{split}_seed{seed}/results.json
fi

exit $EXIT_CODE
"""


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SLURM_SCRIPTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    submitted = []
    errors = []

    for backbone in BACKBONE_CONFIG:
        cfg = BACKBONE_CONFIG[backbone]
        for split in SPLITS:
            for seed in SEEDS:
                split_fold = 0
                job_name = f"cg-{backbone}-{split}-s{seed}"
                script_path = os.path.join(SLURM_SCRIPTS_DIR, f"{job_name}-rorqual.sh")

                script_content = SLURM_TEMPLATE.format(
                    backbone=backbone, split=split, split_fold=split_fold,
                    seed=seed, account=ACCOUNT, time=cfg["time"], mem=cfg["mem"],
                    gpu_type=GPU_TYPE, project=PROJECT, logs_dir=LOGS_DIR,
                    module_loads=MODULE_LOADS, patience=cfg["patience"],
                    batch_size=cfg["batch_size"], extra_args=cfg["extra_args"],
                )

                with open(script_path, "w") as f:
                    f.write(script_content)

                try:
                    result = subprocess.run(
                        ["sbatch", script_path],
                        capture_output=True, text=True, timeout=30,
                    )
                    if result.returncode == 0:
                        job_id = result.stdout.strip().split()[-1]
                        submitted.append((job_name, job_id))
                        print(f"  ✓ {job_name}: Job {job_id}")
                    else:
                        errors.append((job_name, result.stderr.strip()))
                        print(f"  ✗ {job_name}: {result.stderr.strip()}")
                except Exception as e:
                    errors.append((job_name, str(e)))
                    print(f"  ✗ {job_name}: {e}")

    print(f"\n{'='*60}")
    print(f"RORQUAL SUBMISSION SUMMARY")
    print(f"{'='*60}")
    print(f"Submitted: {len(submitted)}/45")
    if errors:
        print(f"Errors: {len(errors)}")
        for name, err in errors:
            print(f"  - {name}: {err}")

    if submitted:
        jobs_file = os.path.join(PROJECT, "rorqual_job_ids.txt")
        with open(jobs_file, "w") as f:
            for name, jid in submitted:
                f.write(f"{name}\t{jid}\n")
        print(f"\nJob IDs saved to: {jobs_file}")

    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
