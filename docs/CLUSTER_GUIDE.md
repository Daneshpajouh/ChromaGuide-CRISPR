# ChromaGuide v2 — DRAC Cluster Guide

Operational guide for running ChromaGuide experiments on Digital Research Alliance of Canada (DRAC) clusters.

## Prerequisites

- DRAC account with `def-kwiese` allocation
- SSH key configured for Alliance clusters
- Duo MFA enrolled (push to mobile device)

## Connecting to Clusters

### SSH Configuration (~/.ssh/config)

```
Host narval narval.alliancecan.ca
    HostName narval.alliancecan.ca
    User amird
    IdentityFile ~/.ssh/alliance_automation
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h:%p
    ControlPersist 48h
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### First Connection (Duo MFA Required)

```bash
ssh narval
# → "Passcode or option (1-1):" → Enter 1 → Approve Duo push
```

Subsequent connections reuse the control socket (no MFA needed) as long as the master connection persists.

## Cluster-Specific Notes

### Narval
```bash
module load StdEnv/2023 python/3.11.5 scipy-stack
```

### Rorqual
```bash
module load StdEnv/2023 python/3.11.5 scipy-stack
```

### Fir
```bash
# SLURM not in default PATH
export PATH=/opt/software/slurm-24.11.6/bin:$PATH

# No module command — use venv directly
# Must specify partition (no default partition):
#SBATCH --partition=gpubase_bygpu_b3   # 1-day limit
#SBATCH --partition=gpubase_bygpu_b4   # 3-day limit
#SBATCH --partition=gpubase_bygpu_b5   # 7-day limit
```

### Nibi
```bash
module load StdEnv/2023 python/3.11.5 scipy-stack
```

### Killarney (Currently Inactive)
```bash
# Requires login shell for SLURM
bash -l -c 'sbatch script.sh'
# SLURM at /cm/shared/apps/slurm/current/bin/
# Partitions: gpubase_h100_b1..b5, gpubase_l40s_b1..b5
# ⚠️ No SLURM account association yet — needs admin setup
```

### Béluga (Currently Inactive)
```bash
# ⚠️ SLURM broken — plugin version mismatch
# cc-tmpfs_mounts.so (23.02.6) incompatible with current sbatch
# Cannot submit jobs until system admins fix this
```

## Working Directory Layout

All clusters use the same layout under `~/scratch/`:

```
~/scratch/chromaguide_v2/           # Main project
├── chromaguide/                    # Source code
├── experiments/                    # Training scripts + SLURM jobs
├── chromaguide/configs/            # YAML configs
├── data/                           # Training data (CSV)
│   ├── train_splitA.csv
│   ├── val_splitA.csv
│   ├── test_splitA.csv
│   └── ... (splits B, C)
└── results/                        # Output directory
    └── <backbone>_split<X>_seed<N>/
        ├── metrics.json
        ├── model_best.pt
        └── slurm-<jobid>.out

~/scratch/chromaguide_v2_env/       # Python virtual environment
```

## Common Operations

### Submit a single job
```bash
cd ~/scratch/chromaguide_v2
sbatch experiments/slurm_jobs/<backbone>_split<X>_seed<N>_<cluster>.sh
```

### Submit all jobs for a cluster
```bash
bash experiments/slurm_jobs/submit_<cluster>.sh
```

### Check job status
```bash
squeue -u $USER
squeue -u $USER -o "%.10i %.12P %.20j %.2t %.10M %.6D %R"
```

### Check completed results
```bash
ls results/*/metrics.json
cat results/<backbone>_split<X>_seed<N>/metrics.json
```

### Cancel all jobs
```bash
scancel -u $USER
```

### View job output
```bash
cat results/<backbone>_split<X>_seed<N>/slurm-<jobid>.out
```

## SLURM Account

| Parameter | Value |
|-----------|-------|
| Account (GPU) | `def-kwiese_gpu` |
| Account (CPU) | `def-kwiese_cpu` |
| User | `amird` |

## Troubleshooting

### "Requested time limit is invalid"
Your partition's max time is lower than `--time`. Check available partitions:
```bash
sinfo -o '%P %l %a'
```
Choose a partition with sufficient time limit.

### "Invalid account or account/partition combination"
The account doesn't exist on this cluster, or isn't associated with the partition. Check:
```bash
sacctmgr show associations where user=$USER format=Account,Partition
```

### SLURM plugin errors (Béluga)
System-level issue. Contact DRAC support or use a different cluster.

### Duo MFA timeout
If you don't approve the push within ~60 seconds, the connection times out. Retry.
