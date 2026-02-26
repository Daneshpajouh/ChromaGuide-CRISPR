#!/usr/bin/env python3
"""Submit all 45 ChromaGuide experiments to Nibi (H100-80GB, 8 GPUs/node)."""
import os
import subprocess
import sys

PROJECT = os.path.expanduser("~/scratch/chromaguide_v2")
OUTPUT_DIR = os.path.join(PROJECT, "results")
LOGS_DIR = os.path.join(PROJECT, "logs")
SLURM_SCRIPTS_DIR = os.path.join(PROJECT, "slurm_scripts")

CLUSTER = "nibi"
ACCOUNT = "def-kwiese_gpu"

BACKBONE_CONFIG = {
    "cnn_gru": {"time": "4:00:00", "mem": "48G", "batch_size": 512, "patience": 15, "extra_args": ""},
    "dnabert2": {"time": "8:00:00", "mem": "64G", "batch_size": 128, "patience": 10, "extra_args": ""},
    "caduceus": {"time": "6:00:00", "mem": "48G", "batch_size": 512, "patience": 15, "extra_args": ""},
    "evo": {"time": "10:00:00", "mem": "64G", "batch_size": 64, "patience": 10, "extra_args": ""},
    "nucleotide_transformer": {"time": "14:00:00", "mem": "80G", "batch_size": 32, "patience": 8, "extra_args": ""},
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
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output={logs_dir}/cg-{backbone}-{split}-s{seed}-%j.out
#SBATCH --error={logs_dir}/cg-{backbone}-{split}-s{seed}-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amir@mystorax.com

echo "===== ChromaGuide v2: {backbone} | Split {split} | Seed {seed} | Nibi ====="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

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

NSAMPLES=$(python -c "import numpy as np; print(len(np.load('data/processed/efficacy.npy')))")
echo "Data samples: $NSAMPLES"
if [ "$NSAMPLES" -lt 100000 ]; then echo "ERROR: Wrong data!"; exit 1; fi

echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"

mkdir -p results/{backbone}_split{split}_seed{seed}

python experiments/train_experiment.py \\
    --backbone {backbone} --split {split} --split-fold {split_fold} --seed {seed} \\
    --data-dir {project}/data --output-dir {project}/results \\
    --no-wandb --patience {patience} --gradient-clip 1.0 --batch-size {batch_size} {extra_args}

EXIT_CODE=$?
echo "Exit: $EXIT_CODE | End: $(date)"
[ -f results/{backbone}_split{split}_seed{seed}/results.json ] && cat results/{backbone}_split{split}_seed{seed}/results.json
exit $EXIT_CODE
"""


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SLURM_SCRIPTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    submitted, errors = [], []

    for backbone in BACKBONE_CONFIG:
        cfg = BACKBONE_CONFIG[backbone]
        for split in SPLITS:
            for seed in SEEDS:
                job_name = f"cg-{backbone}-{split}-s{seed}"
                script_path = os.path.join(SLURM_SCRIPTS_DIR, f"{job_name}-nibi.sh")

                with open(script_path, "w") as f:
                    f.write(SLURM_TEMPLATE.format(
                        backbone=backbone, split=split, split_fold=0,
                        seed=seed, account=ACCOUNT, time=cfg["time"], mem=cfg["mem"],
                        project=PROJECT, logs_dir=LOGS_DIR, module_loads=MODULE_LOADS,
                        patience=cfg["patience"], batch_size=cfg["batch_size"],
                        extra_args=cfg["extra_args"],
                    ))

                try:
                    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True, timeout=30)
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

    print(f"\n{'='*60}\nNIBI SUBMISSION: {len(submitted)}/45")
    if errors:
        print(f"Errors: {len(errors)}")
        for n, e in errors:
            print(f"  - {n}: {e}")

    if submitted:
        with open(os.path.join(PROJECT, "nibi_job_ids.txt"), "w") as f:
            for n, j in submitted:
                f.write(f"{n}\t{j}\n")

    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
