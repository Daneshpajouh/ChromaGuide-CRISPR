# ChromaGuide Experiment Status — Session 6

## Date: February 26, 2026 (Session 6: Real Data Fix)

## CRITICAL FINDING: Synthetic Data Root Cause

### Problem Discovered
All 29 completed experiments (Narval 20/20, Rorqual 9/19) showed **Spearman ρ ≈ 0** (range: -0.004 to 0.014). After investigation, the root cause was identified:

**The training data was synthetic with ZERO correlation between sequences and labels.**

The `preprocess.py` synthetic data generators (`_generate_synthetic_deephf()`, `_generate_synthetic_crispron()`, `_generate_synthetic_fmc()`) produce:
- Random DNA sequences (uniform ACGT sampling)
- Independent random Beta-distributed efficacy scores
- No sequence-efficacy relationship whatsoever

When real CRISPR datasets were not found on the clusters, the code silently fell back to synthetic generation. This made it mathematically impossible for any model to learn.

### Fix Applied
Completely rewrote the data pipeline to use **verified real CRISPR datasets**:

1. **Data source**: CRISPR-FMC benchmark datasets from [xx0220/CRISPR-FMC](https://github.com/xx0220/CRISPR-FMC)
2. **9 datasets, 291,639 sgRNAs total**:
   - Large-scale (Wang et al. 2019): WT (55,603), ESP (58,616), HF (56,887)
   - Medium-scale (Kim et al. 2020): xCas9 (37,738), SpCas9-NG (30,585), Sniper (37,794)
   - Small-scale (Hart 2015/Chuai 2018): HCT116 (4,239), HELA (8,101), HL60 (2,076)
3. **94,615 unique sequences** after cross-dataset deduplication
4. **Efficacy variance = 0.074** (vs. ~0 with synthetic data)

### Files Modified
- `chromaguide/data/acquire.py` — Rewritten with verified download URLs
- `chromaguide/data/preprocess.py` — Rewritten to parse real CSV data, per-dataset normalization
- `experiments/prepare_data.py` — Updated for new pipeline
- `experiments/slurm_data_prep.sh` — Cluster data download script

### Dataset Statistics (Real Data)
| Dataset | N | Efficacy μ ± σ | Cell Line | Cas9 Variant | Source |
|---------|---|----------------|-----------|--------------|--------|
| WT | 55,603 | 0.721 ± 0.222 | HEK293T | WT | Wang 2019 |
| ESP | 58,616 | 0.354 ± 0.189 | HEK293T | eSpCas9 | Wang 2019 |
| HF | 56,887 | 0.474 ± 0.208 | HEK293T | SpCas9-HF1 | Wang 2019 |
| xCas9 | 37,738 | 0.286 ± 0.242 | HEK293T | xCas9 | Kim 2020 |
| SpCas9-NG | 30,585 | 0.400 ± 0.230 | HEK293T | SpCas9-NG | Kim 2020 |
| Sniper | 37,794 | 0.311 ± 0.290 | HEK293T | Sniper-Cas9 | Kim 2020 |
| HCT116 | 4,239 | 0.269 ± 0.182 | HCT116 | WT | Hart 2015 |
| HELA | 8,101 | 0.257 ± 0.182 | HeLa | WT | Hart 2015 |
| HL60 | 2,076 | 0.307 ± 0.171 | HL60 | WT | Wang 2014 |

### Split Statistics (after deduplication to 94,615 unique sequences)
- **Split A (Gene-held-out)**: train=66,445, cal=13,982, test=14,188
- **Split B (Dataset-held-out)**: 9 folds
- **Split C (Cell-line-held-out)**: 4 folds (HEK293T, HCT116, HeLa, HL60)

## Previous Experiment Results (INVALID — Synthetic Data)

### Narval (A100-40GB) — 20/20 COMPLETED (Spearman ≈ 0)
| Job ID | Backbone | Split | Seed | Spearman | Status |
|--------|----------|-------|------|----------|--------|
| 57045459-57045480 | All | All | All | -0.004 to 0.014 | COMPLETED (INVALID) |

### Rorqual (H100-80GB) — 9 COMPLETED, 10 FAILED
- 9 completed: Spearman ≈ 0 (INVALID, same synthetic data issue)
- 3 DNABERT-2 seed=123 FAILED: exit code 1
- 4 Evo FAILED: packages not installed in venv
- 3 NT (redistributed) FAILED: same package issue

### Nibi — 6 jobs (NOT YET CHECKED)
| Job ID | Backbone | Split | Seed | Status |
|--------|----------|-------|------|--------|
| 9334978-9334983 | CNN-GRU | A/B/C | 123/456 | UNKNOWN |

### Fir — NOT YET CHECKED (was DOWN)

## Next Steps (This Session)
1. ✅ Download real CRISPR datasets (9 benchmarks, 291K samples)
2. ✅ Rewrite data pipeline (acquire.py, preprocess.py, prepare_data.py)
3. ✅ Validate pipeline locally (all checks passed)
4. ⬜ Check Nibi and Fir cluster status
5. ⬜ Deploy updated code + real data to all clusters
6. ⬜ Fix Evo/NT package issues on Rorqual
7. ⬜ Resubmit all 45 experiments with real data
8. ⬜ Push to GitHub

## SOTA Comparison Targets
From CRISPR-FMC (Xiang et al. 2025, Frontiers in Genome Editing):
| Dataset | SOTA SCC | SOTA PCC | Method |
|---------|----------|----------|--------|
| WT | 0.861 | 0.889 | CRISPR-FMC |
| ESP | 0.851 | 0.845 | CRISPR-FMC |
| HF | 0.851 | 0.866 | CRISPR-FMC |
| Sniper | 0.935 | 0.957 | CRISPR-FMC |
| HCT116 | ~0.4 | ~0.4 | CRISPR-FMC |
| HELA | ~0.4 | ~0.4 | CRISPR-FMC |
| HL60 | 0.402 | 0.404 | CRISPR-FMC |

**ChromaGuide target**: Spearman ρ ≥ 0.91 (proposal H1 target)

## GitHub
- Branch: `v2-full-rewrite`
- Previous commit: `fe61e96` — "docs: update EXPERIMENT_STATUS.md with full deployment status"
- Pending: Real data pipeline fix + documentation
