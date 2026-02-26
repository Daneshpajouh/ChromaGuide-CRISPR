# ChromaGuide v2 — Experiment Status

> Last updated: 2026-02-26T17:35Z (Session 7)
> Branch: `v2-full-rewrite` | Commit: `121d55f`

## Executive Summary

**180 total GPU jobs** submitted across 4 clusters (Narval, Rorqual, Nibi, Fir) running all 45 experiments on **REAL CRISPR-FMC benchmark data** (291,639 sgRNAs from 9 datasets).

## Root Cause Resolution (Session 6)

Previous experiments (Sessions 1-5) all produced Spearman ≈ 0 because data was synthetic (random uniform noise, variance ≈ 0). This session:
1. Identified 9 real CRISPR-FMC benchmark datasets from [CRISPR-FMC](https://github.com/xx0220/CRISPR-FMC)
2. Rewrote entire data pipeline (`acquire.py`, `preprocess.py`, `prepare_data.py`)
3. Validated: 291,639 sgRNAs, 94,615 unique after dedup, efficacy variance = 0.0743
4. Deployed to all 4 active clusters with verified data integrity

## Data Summary

| Metric | Value |
|--------|-------|
| Total sgRNAs | 291,639 |
| Unique after dedup | 94,615 |
| Datasets | 9 (WT, ESP, HF, xCas9, SpCas9-NG, Sniper, HCT116, HELA, HL60) |
| Cell lines | 4 (HEK293T: 277K, HeLa: 8K, HCT116: 4K, HL60: 2K) |
| Efficacy mean ± std | 0.434 ± 0.273 |
| Efficacy variance | 0.0743 |
| Genes | 2,000 |
| Source | [CRISPR-FMC](https://github.com/xx0220/CRISPR-FMC) |

### Per-Dataset Statistics

| Dataset | Cas9 Variant | Cell Line | N | Mean ± Std |
|---------|-------------|-----------|------|------------|
| WT | SpCas9-WT | HEK293T | 55,603 | 0.721 ± 0.222 |
| ESP | eSpCas9 | HEK293T | 58,616 | 0.354 ± 0.189 |
| HF | SpCas9-HF1 | HEK293T | 56,887 | 0.474 ± 0.208 |
| xCas9 | xCas9 | HEK293T | 37,738 | 0.286 ± 0.242 |
| SpCas9-NG | SpCas9-NG | HEK293T | 30,585 | 0.400 ± 0.230 |
| Sniper | Sniper-Cas9 | HEK293T | 37,794 | 0.311 ± 0.290 |
| HCT116 | SpCas9-WT | HCT116 | 4,239 | 0.269 ± 0.182 |
| HELA | SpCas9-WT | HeLa | 8,101 | 0.257 ± 0.182 |
| HL60 | SpCas9-WT | HL60 | 2,076 | 0.307 ± 0.171 |

### SOTA Benchmarks to Beat (from CRISPR-FMC paper)

| Dataset | CRISPR-FMC SOTA (SCC) | Our Target |
|---------|----------------------|------------|
| WT | 0.861 | ≥ 0.91 |
| ESP | 0.851 | ≥ 0.91 |
| HF | 0.851 | ≥ 0.91 |
| Sniper | 0.935 | ≥ 0.94 |
| HL60 | 0.402 | ≥ 0.50 |

## Experiment Matrix

**5 backbones × 3 splits × 3 seeds = 45 experiments per cluster**

### Backbones

| Backbone | Params | Batch Size | Time (est) | Notes |
|----------|--------|------------|------------|-------|
| CNN-GRU | ~2M | 256-512 | 2-4h | Baseline |
| DNABERT-2 | ~117M | 64-128 | 6-10h | BERT fine-tuning |
| Caduceus | ~7M | 256-512 | 3-6h | Bidirectional Mamba |
| Evo | ~14M | 32-64 | 6-10h | LoRA adapters |
| Nucleotide Transformer | ~500M | 16-32 | 10-18h | Largest model |

### Splits
- **Split A**: Gene-held-out (train=66K, cal=14K, test=14K)
- **Split B** (fold 0): Dataset-held-out, WT test (train=33K, cal=6K, test=56K)
- **Split C** (fold 0): Cell-line-held-out, HEK293T test (train=8K, cal=1.5K, test=85K)

### Seeds: 42, 123, 456

## Cluster Deployment Status

### Narval (A100-40GB) — 45/45 Submitted ✓

| Detail | Value |
|--------|-------|
| Jobs | 57060480 – 57060524 |
| Status | All PENDING (Priority queue) |
| Data | 291,639 samples verified |
| Commit | 121d55f |
| GPU | NVIDIA A100-40GB |

### Rorqual (H100-80GB) — 45/45 Submitted ✓

| Detail | Value |
|--------|-------|
| Jobs | 7388144 – 7388189 |
| Status | All PENDING |
| Data | 291,639 samples verified |
| Commit | 121d55f |
| GPU | NVIDIA H100-80GB |

### Nibi (H100-80GB) — 45/45 Submitted ✓

| Detail | Value |
|--------|-------|
| Jobs | 9358621 – 9358670 |
| Status | 1 RUNNING (DNABERT2 Split A s42), 44 PENDING |
| Data | 291,639 samples verified |
| Commit | 121d55f |
| GPU | NVIDIA H100-80GB |

### Fir (H100-80GB, SFU) — 45/45 Submitted ✓

| Detail | Value |
|--------|-------|
| Jobs | 24668809 – 24668855 |
| Status | All PENDING |
| Data | 291,639 samples verified |
| Commit | 121d55f |
| GPU | NVIDIA H100-80GB (4 per node) |
| Partitions | gpubase_bygpu_b2 (≤12h), gpubase_bygpu_b3 (≤24h) |

### Skipped Clusters

| Cluster | Reason |
|---------|--------|
| Killarney | No SLURM account for def-kwiese |
| Béluga | V100-16GB SLURM incompatibility |

## Thesis Targets

| Target | Metric | Threshold | Status |
|--------|--------|-----------|--------|
| H1: On-target efficacy | Spearman ρ | ≥ 0.91 | AWAITING RESULTS |
| H2: Off-target prediction | AUROC | ≥ 0.92 | Phase 2 (not started) |
| H3: Conformal coverage | Coverage | 90% ± 2% | AWAITING RESULTS |
| Calibration | ECE | < 0.05 | AWAITING RESULTS |
| Significance | p-value | < 0.001 | AWAITING RESULTS |

## Timeline

| Date | Event |
|------|-------|
| Sessions 1-4 | Built ChromaGuide v2 codebase (47 files, 54/54 tests passing) |
| Session 5 | Deployed to 6 clusters, submitted 45 initial jobs |
| Session 6 | Discovered synthetic data root cause, researched real datasets |
| Session 6 (cont) | Rewrote data pipeline, validated locally (291K samples) |
| Session 7 | Deployed to Narval/Rorqual/Nibi/Fir, 180 total jobs submitted |
| Next | Monitor results, collect completed experiments, analyze vs SOTA |

## Files Modified

- `chromaguide/data/acquire.py` — Real CRISPR-FMC download URLs
- `chromaguide/data/preprocess.py` — Per-dataset min-max normalization, pseudo-gene hashing
- `experiments/prepare_data.py` — Complete real data pipeline
- `experiments/submit_all_narval.py` — Narval SLURM submission (A100)
- `experiments/submit_all_rorqual.py` — Rorqual SLURM submission (H100)
- `experiments/submit_all_nibi.py` — Nibi SLURM submission (H100)
- `experiments/submit_all_fir.py` — Fir SLURM submission (H100)
- `docs/DATA_ACQUISITION.md` — Full data provenance
- `docs/EXPERIMENT_STATUS.md` — This file
