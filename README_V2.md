# ChromaGuide v2: Full Thesis Implementation

> **Multi-Modal Deep Learning Framework for CRISPR-Cas9 Guide RNA Design and Efficacy Prediction**
> 
> PhD Thesis — Amir Daneshpajouh, Simon Fraser University
> Supervisor: Dr. Kay C. Wiese

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 ChromaGuide v2                   │
├─────────────────────────────────────────────────┤
│                                                  │
│  sgRNA Sequence ──→ Sequence Encoder ──→ z_s     │
│    (23nt)            (CNN-GRU / DNABERT-2 /      │
│                       Caduceus / Evo)    ∈ ℝ⁶⁴  │
│                                                  │
│  Epigenomic ──→ Epigenomic Encoder ──→ z_e       │
│    (DNase, H3K4me3,  (MLP / CNN-1D)     ∈ ℝ⁶⁴  │
│     H3K27ac)                                     │
│                          ↓                       │
│              ┌───────────────────────┐           │
│              │  Gated Attention      │           │
│              │  Fusion               │           │
│              │  g = σ(W[z_s;z_e]+b)  │           │
│              │  z = [g⊙z_s;(1-g)⊙z_e]│          │
│              └───────────────────────┘           │
│                          ↓                       │
│              ┌───────────────────────┐           │
│              │  Beta Regression Head │           │
│              │  → (μ, φ)             │           │
│              │  → Conformal PI       │           │
│              └───────────────────────┘           │
│                                                  │
│  Off-Target: CNN Scorer + Noisy-OR Aggregation   │
│  Design Score: D(g) = w_e·E - w_r·R - w_u·U     │
├─────────────────────────────────────────────────┤
│  Targets: Spearman ρ ≥ 0.91 | AUROC ≥ 0.92     │
│           Coverage ∈ [88%, 92%] | ECE < 0.05     │
└─────────────────────────────────────────────────┘
```

## Project Structure

```
chromaguide/                    # Main Python package
├── configs/                    # YAML configurations
│   ├── default.yaml            # Base config (256 params)
│   ├── dnabert2.yaml           # DNABERT-2 overrides
│   ├── caduceus.yaml           # Caduceus overrides
│   └── evo.yaml                # Evo adapter overrides
├── data/                       # Data pipeline
│   ├── acquire.py              # Download DeepHF, CRISPRon, CRISPR-FMC, ENCODE
│   ├── preprocess.py           # Parse, normalize, extract epigenomic signals
│   ├── dataset.py              # PyTorch Dataset classes
│   └── splits.py               # Leakage-controlled split construction (A/B/C)
├── modules/                    # Neural network building blocks
│   ├── sequence_encoders.py    # CNN-GRU, DNABERT-2, Caduceus, Evo
│   ├── epigenomic_encoder.py   # MLP / CNN-1D encoder
│   ├── fusion.py               # Gated Attention, Concat-MLP, Cross-Attention, MoE
│   ├── prediction_head.py      # Beta regression (μ, φ)
│   └── conformal.py            # Split conformal prediction + weighted variant
├── models/                     # Full model definitions
│   ├── chromaguide.py          # ChromaGuideModel (on-target)
│   ├── offtarget.py            # Off-target CNN scorer + Noisy-OR
│   └── design_score.py         # Integrated sgRNA design scorer
├── training/                   # Training infrastructure
│   ├── trainer.py              # Full training loop (AMP, early stopping, W&B)
│   ├── losses.py               # Beta NLL, Calibrated Loss, Non-redundancy
│   ├── hpo.py                  # Optuna hyperparameter optimization
│   └── train_offtarget.py      # Off-target training pipeline
├── evaluation/                 # Evaluation & thesis outputs
│   ├── metrics.py              # Spearman, nDCG, ECE, CRPS, AUROC
│   ├── statistical_tests.py    # 5×2cv t-test, BCa bootstrap, Holm-Bonferroni
│   ├── evaluate.py             # Evaluation runner
│   └── thesis_outputs.py       # Publication-quality figures & LaTeX tables
├── utils/                      # Utilities
│   ├── config.py               # OmegaConf config management
│   ├── reproducibility.py      # Seed setting, param counting
│   └── logging.py              # Rich logging setup
└── cli.py                      # Click CLI entry point

scripts_v2/                     # Execution scripts
├── setup_cluster.sh            # One-time SFU Fir cluster setup
├── run_all_experiments.py      # Master SLURM job submitter with dependencies
└── run_5x2cv.py                # 5×2 cross-validation for statistical testing

slurm_v2/                       # SLURM job scripts (SFU Fir, A100-80GB)
├── 01_data_pipeline.sh         # Data acquisition & preprocessing
├── 02_hpo.sh                   # Optuna HPO (50 trials)
├── 03_train_main.sh            # Main training (5 backbones × 3 seeds, array job)
├── 04_ablations.sh             # Ablation studies (12 configs, array job)
├── 05_offtarget.sh             # Off-target module training
├── 06_statistical_tests.sh     # 5×2cv paired t-test
└── 07_thesis_outputs.sh        # Generate figures & tables
```

## Performance Targets

| Metric | Target | Baseline (ChromeCRISPR) |
|--------|--------|------------------------|
| Spearman ρ (on-target) | ≥ 0.91 | 0.876 |
| AUROC (off-target) | ≥ 0.92 | 0.81 (CCLMoff) |
| Conformal coverage | 90% ± 2% | N/A |
| ECE | < 0.05 | N/A |
| p-value (5×2cv) | < 0.001 | — |

## Quick Start

```bash
# Install
pip install -e .

# Download & preprocess data
chromaguide data --stage download
chromaguide data --stage preprocess
chromaguide data --stage splits

# Train (CNN-GRU baseline)
chromaguide train --backbone cnn_gru --split A --seed 42

# Train (DNABERT-2)
chromaguide train --config chromaguide/configs/dnabert2.yaml --split A

# HPO
chromaguide hpo --n-trials 50 --split A

# Off-target training
chromaguide offtarget

# Evaluate
chromaguide evaluate --checkpoint results/checkpoints/cnn_gru_splitA_best.pt --split A

# Generate thesis figures
chromaguide thesis --results-dir results/
```

## SFU Cluster Deployment

```bash
# 1. Setup (one-time)
bash scripts_v2/setup_cluster.sh

# 2. Submit all experiments with dependency management
python scripts_v2/run_all_experiments.py

# Or submit individually:
sbatch slurm_v2/01_data_pipeline.sh
sbatch --dependency=afterok:<DATA_JOB_ID> slurm_v2/02_hpo.sh
sbatch --dependency=afterok:<HPO_JOB_ID> slurm_v2/03_train_main.sh
# ...
```

## Datasets

- **DeepHF**: ~60k sgRNAs, 3 cell lines (HEK293T, HCT116, HeLa), 3 Cas9 variants (WT, ESP, HF)
- **CRISPRon**: 23,902 gRNAs with measured activities
- **CRISPR-FMC**: 9 benchmark datasets
- **ENCODE**: Epigenomic tracks (DNase-seq, H3K4me3, H3K27ac) per cell line
- **GUIDE-seq / CIRCLE-seq**: ~200k off-target pairs

## Splits (Leakage-Controlled)

- **Split A** (primary): Gene-held-out — no gene appears in train and test
- **Split B**: Dataset-held-out (leave-one-dataset-out)
- **Split C**: Cell-line-held-out (leave-one-cell-line-out)

## Statistical Framework

- 5×2cv paired t-test for model comparison (Dietterich 1998)
- BCa bootstrap CIs (10,000 resamples)
- Holm-Bonferroni for primary hypotheses (H1-H3)
- Benjamini-Hochberg for ablation FDR control

## Current Status (Feb 25, 2026)

**Phase 1 — Synthetic Baseline: IN PROGRESS**

45 SLURM jobs submitted across 4 DRAC clusters (Narval, Rorqual, Fir, Nibi).
5 backbones × 3 splits × 3 seeds = 45 experiments on 77,902 synthetic samples.

See [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) for live status.

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) | Job status, IDs, cluster assignments |
| [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md) | Full development chronology |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Technical architecture guide |
| [`docs/CLUSTER_GUIDE.md`](docs/CLUSTER_GUIDE.md) | DRAC cluster connection & operations |

## Estimated Compute

~800-1,200 GPU-hours across DRAC clusters (Narval A100-40GB, Rorqual H100-80GB, Fir A100-80GB, Nibi)
