# ChromaGuide Thesis Implementation Checklist

This document maps each specific target from the PhD proposal/thesis to the corresponding script or source file that implements it.

## 1. Core Objectives & Performance Targets

| Thesis Target | Implementation Script | Source Module | Status |
| :--- | :--- | :--- | :--- |
| **Spearman $\rho \ge 0.911$** (On-target) | `scripts/train_on_real_data_v2.py` | `src/chromaguide/chromaguide_model.py` | 🚀 SUBMITTED (v2) |
| **AUROC $\ge 0.99$** (Off-target) | `scripts/train_off_target_v2.py` | `src/chromaguide/off_target.py` | 🚀 SUBMITTED (v2) |
| **90% Conformal Coverage** | `scripts/calibrate_conformal.py` | `src/chromaguide/conformal.py` | ✅ IMPLEMENTED |
| **Conformal Width $\pm 0.02$** | `scripts/evaluate_all_targets.py` | `src/chromaguide/conformal.py` | ✅ IMPLEMENTED |

## 2. Integrated Designer Score
Score Formula: $S = w_e \mu - w_r R - w_u \sigma$

| Requirement | Implementation Script | Source Module | Status |
| :--- | :--- | :--- | :--- |
| **Formula Implementation** | `scripts/run_designer.py` | `src/chromaguide/design_score.py` | ✅ IMPLEMENTED |
| **Configurable Weights** | `scripts/run_designer.py` | `src/chromaguide/designer.py` | ✅ IMPLEMENTED |
| **Candidate Ranking** | `scripts/run_designer.py` | `src/chromaguide/design_score.py` | ✅ IMPLEMENTED |

## 3. Statistical Rigor
| Requirement | Implementation Script | Details | Status |
| :--- | :--- | :--- | :--- |
| **Bootstrap CI (95%)** | `scripts/run_bootstrap_testing.py` | For Spearman $\rho$ and performance deltas | ✅ IMPLEMENTED |
| **Wilcoxon Signed-Rank** | `scripts/run_bootstrap_testing.py` | Pairwise significance testing | ✅ IMPLEMENTED |
| **Effect Size (Cohen's d)** | `scripts/run_bootstrap_testing.py` | Quantifying magnitude of improvement | ✅ IMPLEMENTED |
| **Significance ($p < 0.001$)** | `scripts/run_bootstrap_testing.py` | Threshold for breakthrough validation | ✅ IMPLEMENTED |

## 4. Ablation Studies
| Ablation Type | Cluster Script (SLURM) | Local Python Script |
| :--- | :--- | :--- |
| **Input Modality** | `scripts/slurm_ablation_modality.sh` | `scripts/run_ablation_modality.py` |
| **Fusion Mechanism** | `scripts/slurm_ablation_fusion.sh` | `scripts/run_ablation_fusion.py` |
| **Backbone (Mamba vs BERT)** | `scripts/slurm_backbone_ablation.sh` | Integrated in `train_on_real_data_v2.py` |

## 5. Architectural Components
| Architecture | Source Module | Logic Path |
| :--- | :--- | :--- |
| **Beta Regression Head** | `src/chromaguide/prediction_head.py` | `BetaRegressionHead` class |
| **Gated Attention Fusion** | `src/chromaguide/fusion.py` | `GatedAttentionFusion` class |
| **Cross-Attention Fusion** | `src/chromaguide/fusion.py` | `CrossAttentionFusion` class |
| **DNABERT-2 / CNN-GRU** | `src/chromaguide/sequence_encoder.py` | `CNNGRUEncoder` class |
| **Epigenomic Integration** | `src/chromaguide/epigenomic_encoder.py` | `EpigenomicEncoder` class |

---
**Note:** All scripts are optimized for the **ChromaGuide** architecture. Legacy scripts (Mamba-only or CRISPRO-specific) should be ignored or archived.
