# ChromaGuide v2 — Experiment Deployment Status

> **Last Updated:** February 25, 2026, 10:30 PM PST  
> **Branch:** `v2-full-rewrite`  
> **Phase:** Synthetic Data Baseline (Phase 1 of 3)

---

## Executive Summary

**45 SLURM jobs** have been submitted across **4 active DRAC clusters** to train all 5 backbone architectures on synthetic CRISPR data. This constitutes Phase 1 (Synthetic Baseline) of the ChromaGuide experiment pipeline. Two clusters (Killarney, Béluga) were unavailable and their jobs were redistributed.

| Metric | Value |
|--------|-------|
| Total Jobs Submitted | **45** |
| Active Clusters | 4 of 6 (Narval, Rorqual, Fir, Nibi) |
| Backbones Under Test | 5 (CNN-GRU, Caduceus, DNABERT-2, NT, Evo) |
| Cross-validation Splits | 3 (A, B, C) |
| Random Seeds per Split | 3 (42, 123, 456) |
| Training Samples | 77,902 synthetic |
| Data Format | CSV (switched from Parquet for cluster compatibility) |

---

## Cluster Status

### Active Clusters

| Cluster | GPU | Jobs | Status | Job IDs |
|---------|-----|------|--------|---------|
| **Narval** | A100-40GB | 12 | Running/Queued | 57042480–57042485, 57042993–57043003 |
| **Rorqual** | H100-80GB | 12 | Queued | 7370937–7370945, 7371320–7371322 |
| **Fir** | A100-80GB | 15 | Running/Queued | 24614338–24614349, 24615398–24615400 |
| **Nibi** | GPU | 6 | Queued | 9330812–9330814, 9331700–9331702 |

### Inactive Clusters

| Cluster | Issue | Resolution |
|---------|-------|------------|
| **Killarney** | No SLURM account association for `amird`. User exists (group `def-kwiese`) but `sacctmgr` shows no associations. The `aip-*` account namespace doesn't include our allocation. | 9 jobs redistributed to Narval (6), Rorqual (3), Fir (3) |
| **Béluga** | SLURM plugin incompatibility: `cc-tmpfs_mounts.so` compiled for SLURM 23.02.6 but sbatch is a different version. System-level issue — cannot be fixed by users. | 6 jobs redistributed to Nibi (3), Narval (3) |

---

## Job Distribution by Backbone

### CNN-GRU (~2M parameters)
Lightweight CNN + Bidirectional GRU baseline.

| Split | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| A | Narval `57042480` | Nibi `9331700` | Nibi `9330812` |
| B | Narval `57042481` | Nibi `9331701` | Nibi `9330813` |
| C | Narval `57042482` | Nibi `9331702` | Nibi `9330814` |

- **Time limit:** 4 hours
- **Memory:** 32 GB
- **Config:** `chromaguide/configs/default.yaml` (CNN-GRU is default)

### Caduceus (~7M parameters)
Bidirectional Mamba architecture for DNA.

| Split | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| A | Narval `57042483` | Narval `57042993` | Narval `57043001` |
| B | Narval `57042484` | Narval `57042995` | Narval `57043002` |
| C | Narval `57042485` | Narval `57042999` | Narval `57043003` |

- **Time limit:** 6 hours
- **Memory:** 64 GB
- **Config:** `chromaguide/configs/caduceus.yaml`

### DNABERT-2 (~117M parameters)
Multi-species genome foundation model with BPE tokenization.

| Split | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| A | Rorqual `7370937` | Rorqual `7371320` | Rorqual `7370938` |
| B | Rorqual `7370939` | Rorqual `7371321` | Rorqual `7370940` |
| C | Rorqual `7370941` | Rorqual `7371322` | Rorqual `7370942` |

- **Time limit:** 12 hours
- **Memory:** 64 GB
- **Config:** `chromaguide/configs/dnabert2.yaml`

### Nucleotide Transformer (~500M parameters)
Large-scale genome language model from InstaDeep.

| Split | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| A | Fir `24614338` | Rorqual `7370943` | Fir `24614339` |
| B | Fir `24614340` | Rorqual `7370944` | Fir `24614341` |
| C | Fir `24614342` | Rorqual `7370945` | Fir `24614343` |

- **Time limit:** 18 hours
- **Memory:** 64 GB
- **Config:** `chromaguide/configs/nucleotide_transformer.yaml`

### Evo (~14M parameters)
Long-context genomic foundation model using StripedHyena.

| Split | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| A | Fir `24614344` | Fir `24615398` | Fir `24614345` |
| B | Fir `24614346` | Fir `24615399` | Fir `24614347` |
| C | Fir `24614348` | Fir `24615400` | Fir `24614349` |

- **Time limit:** 12 hours
- **Memory:** 64 GB
- **Config:** `chromaguide/configs/evo.yaml`

---

## Monitoring Jobs

### Check status on each cluster

```bash
# Narval
ssh narval "squeue -u $USER"

# Rorqual
ssh rorqual "squeue -u $USER"

# Fir (requires special PATH)
ssh fir "export PATH=/opt/software/slurm-24.11.6/bin:\$PATH && squeue -u \$USER"

# Nibi
ssh nibi "squeue -u $USER"
```

### Check completed results

```bash
# On any cluster, from ~/scratch/chromaguide_v2/
ls results/*/metrics.json

# Collect all results
python experiments/collect_results.py
```

### Monitor script

```bash
# Run the monitoring script
bash experiments/monitor_jobs.sh
```

---

## Experiment Details

### Data Pipeline
- **Total samples:** 77,902 synthetic CRISPR guide sequences
- **Features:** 23-nt guide sequence, GC content, thermodynamic features
- **Targets:** Continuous efficacy score (0–1), binary off-target label
- **Splits:** 3-fold gene-held-out cross-validation (A/B/C)
- **Format:** CSV (Parquet caused compatibility issues with some cluster Python environments)

### Training Configuration
- **Optimizer:** AdamW
- **Learning rate:** 1e-4 (CNN-GRU), 2e-5 (foundation models)
- **Scheduler:** CosineAnnealing with warmup
- **Batch size:** 64 (CNN-GRU/Caduceus/Evo), 32 (DNABERT-2/NT)
- **Early stopping:** Patience 10, monitoring validation Spearman ρ
- **Loss:** Huber loss (on-target), Focal loss (off-target)

### Performance Targets (from PhD Proposal)

| Metric | Target | SOTA Reference |
|--------|--------|----------------|
| Spearman ρ (gene-held-out) | ≥ 0.91 | PLM-CRISPR: 0.950 |
| Off-target AUROC | ≥ 0.92 | — |
| Conformal coverage | 90% ± 2% | — |
| ECE | < 0.05 | — |
| Statistical significance | p < 0.001 | — |

---

## Next Steps

### Phase 2: Real Dataset Training
Once synthetic baseline completes:
1. Download real datasets: DeepHF, CRISPRon, ENCODE epigenomic tracks
2. Fine-tune best synthetic checkpoints on real data
3. Evaluate on gene-held-out splits

### Phase 3: Full Evaluation & Thesis
1. Run off-target module training
2. Conformal prediction calibration
3. Integrated design score evaluation
4. Statistical testing (bootstrap, Wilcoxon signed-rank)
5. Ablation studies
6. Generate thesis figures and tables
