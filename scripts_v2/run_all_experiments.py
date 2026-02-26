#!/usr/bin/env python
"""Master experiment runner — submits all SLURM jobs in correct order.

Supports multiple Alliance Canada clusters:
    - narval:    4× A100-40GB, AMD EPYC (Calcul Québec)
    - beluga:    4× V100-16GB, Intel Skylake (→ being replaced by Rorqual)
    - rorqual:   H100-80GB (Calcul Québec, replacement for Béluga)
    - fir:       A100-80GB (SFU)
    - killarney: H100-80GB or L40S-48GB (UofT/Vector, PAICE)

Usage:
    python scripts_v2/run_all_experiments.py --cluster narval
    python scripts_v2/run_all_experiments.py --cluster narval --dry-run
    python scripts_v2/run_all_experiments.py --cluster narval --account def-kwiese
"""
import subprocess
import argparse
import sys
import os
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# Cluster configurations
# ═══════════════════════════════════════════════════════════════

CLUSTER_CONFIGS = {
    "narval": {
        "gpu_type": "a100",
        "gpu_mem": 40,
        "gpu_flag": "--gres=gpu:a100:1",
        "cpus_per_gpu": 12,
        "mem_per_gpu": "125G",
        "partition": None,  # Narval has no named GPU partition
        "module_loads": "module load StdEnv/2023 python/3.11 cuda/12.2",
        "default_account": "def-kwiese",
        "max_time_hr": 24,
        "notes": "AMD EPYC, A100-40GB. Use --gres=gpu:a100:1",
    },
    "beluga": {
        "gpu_type": "v100",
        "gpu_mem": 16,
        "gpu_flag": "--gres=gpu:v100:1",
        "cpus_per_gpu": 10,
        "mem_per_gpu": "46G",
        "partition": None,
        "module_loads": "module load StdEnv/2023 python/3.11 cuda/12.2",
        "default_account": "def-kwiese",
        "max_time_hr": 24,
        "notes": "V100-16GB. Being replaced by Rorqual.",
    },
    "rorqual": {
        "gpu_type": "h100",
        "gpu_mem": 80,
        "gpu_flag": "--gres=gpu:h100:1",
        "cpus_per_gpu": 12,
        "mem_per_gpu": "125G",
        "partition": None,
        "module_loads": "module load StdEnv/2023 python/3.11 cuda/12.2",
        "default_account": "def-kwiese",
        "max_time_hr": 24,
        "notes": "H100-80GB. Béluga replacement.",
    },
    "fir": {
        "gpu_type": "a100",
        "gpu_mem": 80,
        "gpu_flag": "--gres=gpu:a100:1",
        "cpus_per_gpu": 8,
        "mem_per_gpu": "64G",
        "partition": "gpu",
        "module_loads": (
            "source /cvmfs/soft.computecanada.ca/config/profile/bash.sh 2>/dev/null || true\n"
            "module load python/3.11 cuda/12.2"
        ),
        "default_account": "def-kwiese",
        "max_time_hr": 24,
        "notes": "SFU cluster, A100-80GB.",
    },
    "killarney": {
        "gpu_type": "l40s",
        "gpu_mem": 48,
        "gpu_flag": "--gres=gpu:l40s:1",
        "cpus_per_gpu": 16,
        "mem_per_gpu": "128G",
        "partition": None,
        "module_loads": "module load python/3.11 cuda/12.2",
        "default_account": None,  # Requires AIP account
        "max_time_hr": 24,
        "notes": "L40S-48GB (Standard) or H100-80GB (Performance). PAICE cluster.",
    },
    "killarney_h100": {
        "gpu_type": "h100",
        "gpu_mem": 80,
        "gpu_flag": "--gres=gpu:h100:1",
        "cpus_per_gpu": 6,
        "mem_per_gpu": "256G",
        "partition": None,
        "module_loads": "module load python/3.11 cuda/12.2",
        "default_account": None,
        "max_time_hr": 24,
        "notes": "H100-80GB Performance tier on Killarney.",
    },
}


# ═══════════════════════════════════════════════════════════════
# SLURM script generator
# ═══════════════════════════════════════════════════════════════

def generate_slurm_script(
    job_name: str,
    commands: str,
    cluster: str,
    account: str,
    time_hr: int = 24,
    n_gpus: int = 1,
    array: str | None = None,
    mem: str | None = None,
    cpus: int | None = None,
) -> str:
    """Generate a cluster-specific SLURM submission script."""
    cfg = CLUSTER_CONFIGS[cluster]

    mem = mem or cfg["mem_per_gpu"]
    cpus = cpus or cfg["cpus_per_gpu"]
    time_str = f"{time_hr:02d}:00:00"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
    ]

    if cfg["partition"]:
        lines.append(f"#SBATCH --partition={cfg['partition']}")

    lines.extend([
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH {cfg['gpu_flag']}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_str}",
        f"#SBATCH --output=results/logs/{job_name}_%j.out",
        f"#SBATCH --error=results/logs/{job_name}_%j.err",
    ])

    if array:
        lines.append(f"#SBATCH --array={array}")

    lines.extend([
        "",
        f"# Auto-generated for cluster: {cluster}",
        f"# GPU: {cfg['gpu_type'].upper()}-{cfg['gpu_mem']}GB",
        f"# {cfg['notes']}",
        "",
        "echo \"Job $SLURM_JOB_ID on $(hostname) started at $(date)\"",
        f"echo \"Cluster: {cluster} | GPU: {cfg['gpu_type']}\"",
        "",
        "# Load modules",
        cfg["module_loads"],
        "",
        "# Activate environment",
        "source $HOME/chromaguide_v2_env/bin/activate",
        "cd $HOME/ChromaGuide-CRISPR",
        "",
        "mkdir -p results/logs results/checkpoints",
        "",
        commands,
        "",
        "echo \"Job $SLURM_JOB_ID finished at $(date)\"",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Job definitions
# ═══════════════════════════════════════════════════════════════

def get_job_definitions(cluster: str) -> list[dict]:
    """Define all experiment jobs."""
    cfg = CLUSTER_CONFIGS[cluster]

    # Adjust batch sizes based on GPU memory
    if cfg["gpu_mem"] <= 16:
        batch_size = 64   # V100-16GB
    elif cfg["gpu_mem"] <= 40:
        batch_size = 128  # A100-40GB
    else:
        batch_size = 256  # A100-80GB / H100-80GB

    jobs = [
        # Phase 1: Data
        {
            "name": "cg-data",
            "time_hr": 4,
            "n_gpus": 0,
            "cpus": 8,
            "mem": "32G",
            "commands": """
echo "=== Data Acquisition & Preprocessing ==="
python -m chromaguide.cli data --stage download
python -m chromaguide.cli data --stage preprocess
python -m chromaguide.cli data --stage splits
echo "Data pipeline complete"
""",
            "depends_on": None,
        },
        # Phase 2: HPO
        {
            "name": "cg-hpo",
            "time_hr": 24,
            "commands": f"""
echo "=== Hyperparameter Optimization ==="
python -m chromaguide.cli hpo --n-trials 50 --split A
echo "HPO complete"
""",
            "depends_on": "cg-data",
        },
        # Phase 3: Main training (array job: 5 backbones × 3 seeds)
        {
            "name": "cg-train",
            "time_hr": 24,
            "array": "0-14",
            "commands": f"""
BACKBONE_NAMES=(cnn_gru dnabert2 nucleotide_transformer caduceus evo)
SEED_VALUES=(42 123 456)

BACKBONE_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

BACKBONE=${{BACKBONE_NAMES[$BACKBONE_IDX]}}
SEED=${{SEED_VALUES[$SEED_IDX]}}

echo "=== Training: $BACKBONE, seed=$SEED ==="

python -m chromaguide.cli train \\
    --backbone $BACKBONE \\
    --seed $SEED \\
    --split A \\
    --wandb \\
    training.batch_size={batch_size}

# Cross-split evaluation
CKPT="results/checkpoints/${{BACKBONE}}_splitA_best.pt"
if [ -f "$CKPT" ]; then
    for SPLIT in B C; do
        python -m chromaguide.cli evaluate --checkpoint $CKPT --split $SPLIT
    done
fi
""",
            "depends_on": "cg-hpo",
        },
        # Phase 4: Ablations (array job: 12 configs)
        {
            "name": "cg-ablation",
            "time_hr": 12,
            "array": "0-11",
            "commands": f"""
ABLATION_NAMES=(
    seq_only accessibility_only histone_only full_epi mismatched_cell
    fusion_concat fusion_gated fusion_cross fusion_moe
    no_mod_dropout no_cal_loss mse_loss
)
OVERRIDES_LIST=(
    "model.modality_dropout.prob=1.0"
    "data.epigenomic.tracks=[DNase]"
    "data.epigenomic.tracks=[H3K4me3,H3K27ac]"
    ""
    ""
    "model.fusion.type=concat_mlp"
    "model.fusion.type=gated_attention"
    "model.fusion.type=cross_attention"
    "model.fusion.type=moe"
    "model.modality_dropout.enabled=false"
    "training.loss.lambda_cal=0.0"
    "training.loss.primary=mse"
)

NAME=${{ABLATION_NAMES[$SLURM_ARRAY_TASK_ID]}}
OVERRIDES=${{OVERRIDES_LIST[$SLURM_ARRAY_TASK_ID]}}

echo "=== Ablation: $NAME ==="

python -m chromaguide.cli train \\
    --backbone cnn_gru --seed 42 --split A --wandb \\
    training.batch_size={batch_size} \\
    $OVERRIDES
""",
            "depends_on": "cg-hpo",
        },
        # Phase 5: Off-target
        {
            "name": "cg-offtarget",
            "time_hr": 8,
            "commands": f"""
echo "=== Off-Target Module Training ==="
python -m chromaguide.cli offtarget
""",
            "depends_on": "cg-data",
        },
        # Phase 6: Statistical tests
        {
            "name": "cg-stats",
            "time_hr": 48,
            "commands": f"""
echo "=== 5x2 Cross-Validation & Statistical Tests ==="
python scripts_v2/run_5x2cv.py
""",
            "depends_on": "cg-train",
        },
        # Phase 7: Thesis outputs
        {
            "name": "cg-thesis",
            "time_hr": 2,
            "n_gpus": 0,
            "cpus": 4,
            "mem": "16G",
            "commands": """
echo "=== Generating Thesis Figures & Tables ==="
python -m chromaguide.cli thesis --results-dir results/
echo "Outputs in: results/figures/ and results/tables/"
""",
            "depends_on": "cg-train,cg-ablation,cg-offtarget,cg-stats",
        },
    ]

    return jobs


# ═══════════════════════════════════════════════════════════════
# Submission logic
# ═══════════════════════════════════════════════════════════════

def submit_job(script_path: str, dependency: str | None, dry_run: bool) -> str:
    """Submit SLURM job, return job ID."""
    cmd = ["sbatch"]
    if dependency:
        cmd.extend(["--dependency", dependency])
    cmd.append(script_path)

    label = f"[DRY] " if dry_run else ""
    print(f"  {label}sbatch {' '.join(cmd[1:])}")

    if dry_run:
        return f"DRY_{Path(script_path).stem}"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        sys.exit(1)

    job_id = result.stdout.strip().split()[-1]
    print(f"    → Job ID: {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser(description="ChromaGuide v2: Master Experiment Runner")
    parser.add_argument("--cluster", required=True, choices=list(CLUSTER_CONFIGS.keys()),
                        help="Target cluster")
    parser.add_argument("--account", default=None, help="SLURM account (e.g., def-kwiese)")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts only, don't submit")
    parser.add_argument("--output-dir", default="slurm_v2/generated", help="Where to write generated scripts")
    args = parser.parse_args()

    account = args.account or CLUSTER_CONFIGS[args.cluster]["default_account"]
    if account is None:
        print("ERROR: No default account for this cluster. Provide --account.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = CLUSTER_CONFIGS[args.cluster]

    print("=" * 60)
    print(f"ChromaGuide v2: Experiment Submission")
    print(f"Cluster:  {args.cluster}")
    print(f"GPU:      {cfg['gpu_type'].upper()}-{cfg['gpu_mem']}GB")
    print(f"Account:  {account}")
    print(f"Dry run:  {args.dry_run}")
    print("=" * 60)

    jobs = get_job_definitions(args.cluster)
    job_ids = {}

    for job in jobs:
        # Generate script
        gpu_flag_override = None
        if job.get("n_gpus", 1) == 0:
            # CPU-only job: remove GPU flag from generated script
            pass

        script_content = generate_slurm_script(
            job_name=job["name"],
            commands=job["commands"],
            cluster=args.cluster,
            account=account,
            time_hr=job.get("time_hr", 24),
            array=job.get("array"),
            mem=job.get("mem"),
            cpus=job.get("cpus"),
        )

        script_path = output_dir / f"{job['name']}_{args.cluster}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

        # Build dependency string
        dep_str = None
        if job["depends_on"]:
            dep_names = [d.strip() for d in job["depends_on"].split(",")]
            dep_ids = [job_ids[d] for d in dep_names if d in job_ids]
            if dep_ids:
                dep_str = "afterok:" + ":".join(dep_ids)

        print(f"\n--- {job['name']} ---")
        job_id = submit_job(str(script_path), dep_str, args.dry_run)
        job_ids[job["name"]] = job_id

    print("\n" + "=" * 60)
    print("All jobs submitted!")
    for name, jid in job_ids.items():
        print(f"  {name}: {jid}")
    print("=" * 60)
    print(f"\nGenerated scripts: {output_dir}/")
    print("Monitor: squeue -u $USER")
    print("Cancel all: scancel -u $USER")


if __name__ == "__main__":
    main()
