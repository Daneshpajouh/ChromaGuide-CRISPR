# ChromaGuide v2 — Development Log

> Chronological record of all development, deployment, and experiment activities.  
> **Branch:** `v2-full-rewrite`

---

## Session 1: v2 Full Rewrite (Feb 25, 2026)

### Codebase Architecture
Rewrote the entire ChromaGuide framework from scratch as a clean, modular Python package:

```
chromaguide/
├── __init__.py
├── cli.py                          # CLI entry point
├── configs/
│   ├── __init__.py
│   ├── default.yaml                # CNN-GRU (default backbone)
│   ├── caduceus.yaml               # Caduceus config
│   ├── dnabert2.yaml               # DNABERT-2 config
│   ├── evo.yaml                    # Evo config
│   └── nucleotide_transformer.yaml # NT config
├── data/
│   ├── __init__.py
│   ├── acquire.py                  # Dataset download/cache
│   ├── dataset.py                  # PyTorch Dataset class
│   ├── preprocess.py               # Feature engineering
│   └── splits.py                   # Gene-held-out CV splits
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py                 # Main evaluation loop
│   ├── metrics.py                  # Spearman, AUROC, ECE, etc.
│   ├── statistical_tests.py        # Bootstrap CI, Wilcoxon
│   └── thesis_outputs.py           # LaTeX tables, figures
├── models/
│   ├── __init__.py
│   ├── chromaguide.py              # Main model (3-module)
│   ├── design_score.py             # Integrated design scorer
│   └── offtarget.py                # Off-target prediction head
├── modules/
│   ├── __init__.py
│   ├── conformal.py                # Conformal prediction
│   ├── epigenomic_encoder.py       # Epigenomic feature encoder
│   ├── fusion.py                   # Cross-attention fusion
│   ├── prediction_head.py          # MLP prediction heads
│   └── sequence_encoders.py        # All 5 backbone encoders
├── training/
│   ├── __init__.py
│   ├── hpo.py                      # Hyperparameter optimization
│   ├── losses.py                   # Huber + Focal loss
│   ├── train_offtarget.py          # Off-target training loop
│   └── trainer.py                  # Main training loop
└── utils/
    ├── __init__.py
    ├── config.py                   # OmegaConf-based config
    ├── logging.py                  # Structured logging
    └── reproducibility.py          # Seed setting, determinism
```

**Total:** 47 Python files implementing the complete ChromaGuide framework.

### Key Decisions
- Used OmegaConf for YAML-based configuration (cleaner than argparse)
- Sequence encoders are modular — swappable via config `backbone` field
- Added NucleotideTransformerEncoder (was missing from v1)
- Gene-held-out cross-validation ensures no data leakage between splits

---

## Session 2: Experiment Infrastructure (Feb 25, 2026)

### Training Script
Created `experiments/train_experiment.py` — master training script that:
1. Loads config from YAML
2. Prepares data splits
3. Initializes model with specified backbone
4. Trains with early stopping
5. Evaluates on held-out test set
6. Saves metrics.json with all results

### SLURM Job Generation
Created `experiments/generate_slurm_jobs.py` to generate all 45 SLURM scripts:
- 5 backbones × 3 splits × 3 seeds = 45 experiments
- Scripts tailored per cluster (GPU type, time limits, modules)
- Organized under `experiments/slurm_jobs/`

### Data Preparation
Created `experiments/prepare_data.py`:
- Generates 77,902 synthetic CRISPR guide sequences
- Features: 23-nt sequences, GC content, thermodynamic features
- Labels: continuous efficacy (0–1), binary off-target
- Outputs: CSV format (switched from Parquet for compatibility)
- Pre-split into A/B/C gene-held-out folds

---

## Session 3: Cluster Deployment (Feb 25–26, 2026)

### Cluster Access Setup
- SSH key: `~/.ssh/alliance_automation`
- SSH config with ControlMaster for persistent connections
- Duo Push MFA (option 1: "Amir's iPhone") required per cluster
- Expect scripts for automated Duo handling

### Deployment Steps (All 6 clusters)
1. Established SSH with Duo MFA
2. Created `~/scratch/chromaguide_v2/` working directory
3. Created Python venv at `~/scratch/chromaguide_v2_env/`
4. Installed dependencies: torch, omegaconf, pyarrow→CSV fallback
5. Synced full codebase via SCP
6. Verified imports (32/32 passing on all clusters)
7. Ran data preparation (77,902 samples)

### Issues Encountered & Resolved

#### 1. PyArrow Incompatibility
**Problem:** `pyarrow` failed to install on some clusters due to old glibc.  
**Solution:** Switched data format from Parquet to CSV. Updated `prepare_data.py` and `train_experiment.py` to read CSV. Both formats supported as fallback.  
**Commit:** `af793ae`

#### 2. Fir — sbatch Not in PATH
**Problem:** Default PATH on Fir doesn't include SLURM binaries.  
**Solution:** Found sbatch at `/opt/software/slurm-24.11.6/bin/sbatch`. Added `export PATH=/opt/software/slurm-24.11.6/bin:$PATH` to Fir scripts.

#### 3. Fir — Time Limit Exceeded
**Problem:** All 12 Fir jobs failed with "Requested time limit is invalid". Default partition `cpubase_bynode_b1` has 3h limit.  
**Solution:** Added `#SBATCH --partition=gpubase_bygpu_b3` (1-day limit) to all Fir scripts. Resubmitted successfully.

#### 4. Killarney — No SLURM Account
**Problem:** `sacctmgr show associations where user=amird` returns empty. User exists in group `def-kwiese` but has no SLURM account association. Killarney uses `aip-*` namespace.  
**Status:** Cannot submit jobs. Likely needs admin setup.  
**Workaround:** Redistributed 9 jobs to Narval (6), Rorqual (3), Fir (3).

#### 5. Béluga — SLURM Plugin Incompatibility
**Problem:** `cc-tmpfs_mounts.so` compiled for SLURM 23.02.6 but sbatch binary is a different version. Error: "Incompatible Slurm plugin". All sbatch commands fail.  
**Status:** System-level issue — cannot be fixed by users.  
**Workaround:** Redistributed 6 jobs to Nibi (3), Narval (3).

#### 6. Killarney — sbatch Requires Login Shell
**Problem:** `bash -c 'sbatch ...'` fails because SLURM not in default PATH.  
**Solution:** Use `bash -l -c 'sbatch ...'` to source login profile. sbatch located at `/cm/shared/apps/slurm/current/bin/sbatch`.

---

## Job Submission Timeline

### Feb 25, 2026 — ~10:00 PM PST
- Submitted **6 jobs** on **Narval** (CNN-GRU seed42 × 3, Caduceus seed42 × 3)
  - Job IDs: 57042480–57042485
  - Status: Running

### Feb 25, 2026 — ~10:05 PM PST
- Submitted **9 jobs** on **Rorqual** (DNABERT-2 seed42/456 × 3, NT seed123 × 3)
  - Job IDs: 7370937–7370945
  - Status: Queued (Priority)

### Feb 25, 2026 — ~10:20 PM PST
- Attempted Fir — failed due to time limit
- Fixed partition → `gpubase_bygpu_b3`
- Resubmitted **12 jobs** on **Fir** (NT seed42/456 × 6, Evo seed42/456 × 6)
  - Job IDs: 24614338–24614349
  - Status: 2 Running (NT A/B on fc10602), 10 Queued

### Feb 25, 2026 — ~10:22 PM PST
- Submitted **3 jobs** on **Nibi** (CNN-GRU seed456 × 3)
  - Job IDs: 9330812–9330814
  - Status: Queued

### Feb 25, 2026 — ~10:23 PM PST
- Attempted Béluga — SLURM plugin error (not fixable)
- Attempted Killarney — no SLURM account
- Decision: Redistribute 15 failed jobs

### Feb 25, 2026 — ~10:25 PM PST
- Generated 15 redistributed SLURM scripts
- Submitted **3 extra jobs** on **Fir** (Evo seed123 × 3)
  - Job IDs: 24615398–24615400

### Feb 25, 2026 — ~10:27 PM PST
- Submitted **3 extra jobs** on **Nibi** (CNN-GRU seed123 × 3)
  - Job IDs: 9331700–9331702

### Feb 25, 2026 — ~10:28 PM PST
- Submitted **6 extra jobs** on **Narval** (Caduceus seed123/456 × 6)
  - Job IDs: 57042993–57043003

### Feb 25, 2026 — ~10:29 PM PST
- Submitted **3 extra jobs** on **Rorqual** (DNABERT-2 seed123 × 3)
  - Job IDs: 7371320–7371322

**Total: 45 jobs submitted across 4 clusters.**

---

## Git History (v2-full-rewrite branch)

| Commit | Date | Description |
|--------|------|-------------|
| `af793ae` | Feb 25 | Fix: use CSV instead of parquet, support both formats |
| `7b84e2a` | Feb 25 | Add experiment infrastructure: training scripts, SLURM jobs, NT encoder |
| `996bdfa` | Feb 25 | fix: whitelist chromaguide/data/ and chromaguide/models/ in .gitignore |
| `d70c3be` | Feb 25 | feat: ChromaGuide v2 full rewrite — complete framework implementation |
| `c220ceb` | earlier | FIX: Use CVMFS profile for module initialization on FIR compute nodes |

---

## Architecture: 5 Sequence Encoders

### 1. CNN-GRU (Default Baseline, ~2M params)
- 3-layer 1D CNN (128→256→512 filters) with BatchNorm + ReLU
- 2-layer Bidirectional GRU (hidden=256)
- Output: 512-dim sequence embedding
- Fastest to train (~1–2 hours)

### 2. Caduceus (~7M params)
- Bidirectional Mamba (state-space model) for DNA
- Pre-trained: `kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16`
- Fine-tuned with LoRA (rank=8)
- Output: 256-dim embedding

### 3. DNABERT-2 (~117M params)
- BERT architecture with BPE tokenization for multi-species genomes
- Pre-trained: `zhihan1996/DNABERT-2-117M`
- Fine-tuned with LoRA (rank=16)
- Output: 768-dim embedding

### 4. Nucleotide Transformer (~500M params)
- Large-scale genome language model from InstaDeep
- Pre-trained: `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`
- Fine-tuned with LoRA (rank=16)
- Output: 1024-dim embedding

### 5. Evo (~14M params)
- Long-context model using StripedHyena architecture
- Pre-trained: `togethercomputer/evo-1-131k-base`
- Fine-tuned with LoRA (rank=8)
- Output: 512-dim embedding

---

## Cluster Technical Reference

### Narval (narval.alliancecan.ca)
- **GPU:** NVIDIA A100-40GB
- **SLURM:** Standard (`sbatch` in PATH)
- **Account:** `def-kwiese_gpu`
- **Partition:** Default

### Rorqual (rorqual.alliancecan.ca)
- **GPU:** NVIDIA H100-80GB
- **SLURM:** Standard
- **Account:** `def-kwiese_gpu`
- **Partition:** Default

### Fir (fir.alliancecan.ca)
- **GPU:** NVIDIA A100-80GB
- **SLURM:** `/opt/software/slurm-24.11.6/bin/sbatch`
- **Account:** `def-kwiese_gpu`
- **Partition:** `gpubase_bygpu_b3` (1-day limit)
- **Note:** No `module` command; uses non-login shell by default

### Nibi (nibi.alliancecan.ca)
- **GPU:** Various
- **SLURM:** Standard
- **Account:** `def-kwiese_gpu`
- **Partition:** Default

### Killarney (killarney.alliancecan.ca) — INACTIVE
- **GPU:** H100, L40S
- **SLURM:** `/cm/shared/apps/slurm/current/bin/sbatch` (login shell required)
- **Issue:** No SLURM account association for user `amird`
- **Partitions available:** `gpubase_h100_b1..b5`, `gpubase_l40s_b1..b5`

### Béluga (beluga.alliancecan.ca) — INACTIVE
- **GPU:** NVIDIA V100-16GB
- **SLURM:** Broken (plugin incompatibility)
- **Issue:** `cc-tmpfs_mounts.so` version mismatch
