# SOTA Audit And Benchmark Master Plan (2026-03-02)

## Purpose

This is the branch-level audit document for four things:

1. What has actually been built, run, and measured in this repository.
2. What is currently proven under our own proposal conditions.
3. What the latest externally published on-target and off-target benchmark landscape looks like as of 2026-03-02.
4. What still must be done before we can honestly claim that we outperform the latest state of the art across models, datasets, and evaluation regimes.

This document is intentionally strict. It separates:

- internal optimization success,
- claim-valid branch results,
- external literature benchmarks,
- and still-unproven extrapolations.

## Executive Conclusion

As of 2026-03-02:

- We have **not yet** honestly demonstrated that this branch outperforms all current external SOTA models across all relevant datasets and metrics.
- We have **not yet** fully met the non-stacked on-target proposal threshold under the exact internal claim conditions.
- We **have** exceeded several other proposal components, including stacked/OOF on-target, off-target held-out performance, conformal coverage, integrated ranking, and significance checks.
- The remaining hard blocker is **on-target non-stacked performance under the branch's Split A claim regime**, where we are still short by about `1.10e-4`.

The correct current answer is therefore:

- Internal progress: strong.
- Proposal fully met under non-stacked claim conditions: **no, not yet**.
- External "we beat all latest SOTA" claim: **no, not yet claimable**.

## Current Verified Branch Status

### On-Target (Internal Proposal Conditions)

Target and baseline:

- Proposal non-stacked target (`Split A`, claim-valid): `>= 0.911` Spearman
- Baseline used in proposal comparisons: `0.876`

Current verified bests:

- Best stacked/OOF single:
  - `0.9700928819028961`
  - artifact family: `results/runs/w33_oof_hgbr_k240_d8_i300_lr004_s21`

- Best stacked/OOF ensemble:
  - `0.9702816198275191`
  - artifact family: `w32`/`w33` OOF stackers

- Best non-stacked single:
  - `0.8892076527146103`
  - artifact: `results/runs/w50opt_nibi_w6_t0065.json`

- Best non-stacked ensemble (current valid frontier):
  - `0.9108903031450908`
  - artifact: `results/runs/w55b_exact_tiny_alpha_full90_summary.json`
  - predictions: `results/runs/w55b_exact_tiny_alpha_full90_predictions.csv`

Remaining gap:

- Gap to `0.911`: `0.00010969685490924697`
- Delta vs `0.876`: `+0.034890303145090784`
- Required delta target: `>= 0.035`
- Remaining delta gap: `0.000109696854909216`

Conclusion:

- The non-stacked claim-valid ensemble is a near miss, but it is still a miss.
- We must not round this up to a pass.

### Strict Split-A Regime

- Best strict Split A single:
  - `0.4024609009171111`
  - artifact: `results/runs/strictA_ng20000_opt_cnn_d512_gate_s42.json`

Conclusion:

- We do not meet any 0.911-style target under the strict gene-held-out regime.

### Off-Target (Matched Held-Out Test)

Matched held-out test metrics already stored in branch status:

- Guide-only baseline:
  - AUROC: `0.9526329010166713`
  - AUPRC: `0.9996944435180574`

- Pair-aware model:
  - AUROC: `0.9999832097198535`
  - AUPRC: `0.9999999193805115`

- Mean AUROC gain vs guide-only:
  - `+0.047093894625403324`

- Statistical result:
  - `p(delta <= 0) = 0.0`

Optuna off-target tuning status:

- `nibi` tuned best summary: `best_auroc = 1.0`, `best_auprc = 1.0`
- `rorqual` tuned best summary: `best_auroc = 1.0`, `best_auprc = 1.0`

Operational conclusion:

- On the branch's current off-target benchmark, additional GPU tuning is low-yield because the observed tuned objective is saturated.

### Calibration, Integrated Ranking, And Significance

- Conformal empirical coverage:
  - `0.9034172661870503`
  - target band: `0.90 +/- 0.02`
  - status: pass

- Integrated ranking vs component baselines:
  - stored as pass in `results/runs/w49_proposal_status_latest.json`
  - status: pass

- On-target significance vs baseline:
  - stored as pass in `results/runs/w49_proposal_status_latest.json`
  - status: pass

### Branch-Level Pass / Fail Snapshot

Stored in `results/runs/w49_proposal_status_latest.json`:

- On-target non-stacked single `>= 0.911`: FAIL
- On-target non-stacked ensemble `>= 0.911`: FAIL
- On-target non-stacked delta vs `0.876` `>= 0.035`: FAIL
- On-target stacked/OOF `>= 0.911`: PASS
- Strict Split-A best single `>= 0.911`: FAIL
- Off-target pair-aware AUROC `>= 0.92`: PASS
- Off-target pair-aware AUROC `>= 0.99`: PASS
- Off-target improvement significance `p < 0.001`: PASS
- Conformal coverage within `0.90 +/- 0.02`: PASS
- Integrated NDCG@20 / Top20 improvement: PASS
- On-target significance `p < 0.001`: PASS

Global internal conclusion:

- `proposal_targets_all_met_nonstacked_claim = false`
- `external_sota_outperformed_claimable = false`

## Repository-Wide Branch Audit

This repository currently has only two real tracked local branches plus their matching remotes:

- `Dev`
- `main`
- `origin/Dev`
- `origin/main`

All four refs currently point to the **same commit**:

- commit `8ac2ab3`
- subject: `v6 training results and evaluation: multimodal (d_model=64, cross-attention) + off-target (residual net ensemble)`

That means there is **not** a divergent branch landscape to reconcile. The correct audit model is:

1. one tracked branch-head snapshot (`8ac2ab3`), and
2. one newer, much larger **uncommitted working-tree state** containing the current extended experiments, scripts, and status artifacts.

This distinction matters because the tracked branch-head snapshot contains much older and much weaker metrics than the current working tree.

### Tracked Branch-Head Snapshot (`8ac2ab3`)

The tracked branch-head commit includes the following core report and metric artifacts:

- `CRITICAL_FIX_SUMMARY.md`
- `DEPLOYMENT_REPORT_FINAL.md`
- `TRAINING_RESULTS_SUMMARY.md`
- `multimodal_metrics.json`
- `on_target_metrics.json`
- `sequence_baseline_metrics.json`
- `sequence_only_metrics.json`
- `results/completed_jobs/ablation_fusion_results.json`
- `results/completed_jobs/ablation_modality_results.json`
- `results/completed_jobs/final_comparison.csv`
- `results/completed_jobs/final_comparison.json`
- `results/completed_jobs/hpo_optuna_results.json`
- `results/completed_jobs/mamba_variant_results.json`
- `results/completed_jobs/results_summary.csv`
- `results/experiment_results_table.csv`
- `test_results_on_target_cnn_gru.csv`
- `designer_demo_results.csv`

### Tracked Branch-Head Results (What `8ac2ab3` Actually Shows)

These are the meaningful branch-head metrics directly visible in the tracked artifact files.

#### Legacy On-Target Metrics Stored In Branch Head

- `sequence_baseline_metrics.json`
  - best validation rho: `0.8237872762428666`
  - gold/test rho: `0.8426924470180335`

- `sequence_only_metrics.json`
  - best validation rho: `0.8201507188718015`
  - gold/test rho: `0.8428699338675333`

- `multimodal_metrics.json`
  - best validation rho: `0.8394356385023982`
  - gold rho: `0.8540807303917752`

- `on_target_metrics.json`
  - best validation rho: `0.8374460742397701`
  - test rho: `0.8522973655009756`

Interpretation:

- The tracked branch-head snapshot was an earlier generation where on-target results were in the `0.84` to `0.85` test range.
- Those results are materially weaker than the current working-tree frontier (`0.9108903031450908` non-stacked ensemble).

#### Legacy Off-Target / Ablation / HPO Results Stored In Branch Head

From `results/completed_jobs/results_summary.csv` and related JSON files:

- `mamba_variant`
  - test Spearman rho: `-0.0718641755659009`
  - p-value: `0.31936110957591635`

- `ablation_fusion` best listed test result
  - concatenation: `0.039411931479662904`
  - gated attention: `0.001103655705439798`
  - cross-attention: `-0.01339425490913125`

- `ablation_modality`
  - sequence-only: `-0.01528928845845677`
  - multimodal: `-0.05502253742198189`

- `hpo_optuna_results.json`
  - best validation rho: `0.1389748367428169`
  - test rho: `-0.011836152736745651`
  - trials: `50`

Interpretation:

- The tracked branch-head contains multiple older synthetic or unstable evaluation outputs that are not representative of the current branch state.
- These legacy artifacts should be preserved as historical evidence, but they must **not** be confused with the current claim-valid benchmark frontier.

### Historical Documents In Branch Head

The branch-head markdown reports document the earlier phase of work:

- `CRITICAL_FIX_SUMMARY.md`
  - documents the DNABERT-2 loading fix and the early thesis-era expectation that this would unlock `rho >= 0.911`

- `DEPLOYMENT_REPORT_FINAL.md`
  - documents deployment to Narval, the DNABERT-2 fix, focal-loss off-target training, and expected outcomes

- `TRAINING_RESULTS_SUMMARY.md`
  - records an earlier state where:
    - off-target best AUROC reached `0.8042`
    - multimodal on-target reached `0.8523`

Interpretation:

- These documents are historically useful and should stay in the audit.
- They are not the final state of the current workstream.
- They describe a much earlier stage, before the larger benchmark rebuild, cluster sweeps, Optuna waves, and non-stacked ensemble refinement series.

### Current Working-Tree Additions (Not Yet Committed At Branch Head)

The working tree now contains substantial newer artifacts not present in `8ac2ab3`, including:

- `LATEST_STATUS_2026-02-26.md`
- `RUN_LOG_2026-02-25_2026-02-26.md`
- `results/runs/w49_proposal_status_latest.json`
- `results/runs/w51_runtime_snapshot.json`
- `results/runs/w55b_exact_tiny_alpha_full90_summary.json`
- `results/runs/w55b_exact_tiny_alpha_full90_predictions.csv`
- `scripts/train_on_target_trainval.py`
- `scripts/train_on_target_ranked.py`
- `scripts/train_on_target_deephf_style.py`
- `scripts/optuna_tune_on_target.py`
- `scripts/optuna_tune_off_target.py`
- `scripts/optimize_nonstacked_ensemble.py`
- multiple `w50` / `w51` / `w52` / `w53` / `w55` run artifacts

These files contain the current, materially stronger benchmark state summarized elsewhere in this document.

### Audit Rule For This Repository

For any future report, use this rule:

- tracked branch-head artifacts (`8ac2ab3`) = historical baseline / legacy branch snapshot
- current working-tree artifacts in `results/runs/` and the new status documents = current active benchmark truth

Do not merge those two layers into one undifferentiated result summary.

### Companion Inventory Deliverables

To make the repo-wide audit fully inspectable, this document is now paired with two generated inventory artifacts:

- `AUDIT_ARTIFACT_INVENTORY_2026-03-02.json`
- `AUDIT_ARTIFACT_APPENDIX_2026-03-02.md`
- `EXTERNAL_SOTA_BENCHMARK_MATRIX_2026-03-02.json`
- `PUBLIC_SOTA_TARGETS_2026-03-02.json`
- `PUBLIC_SOTA_GAP_ANALYSIS_2026-03-02.json`
- `PUBLIC_ON_TARGET_READINESS_2026-03-02.json`
- `REMAINING_RESEARCH_BROWSER_PROMPT_2026-03-02.md`

These were generated directly from the current working tree and index:

- total indexed artifacts: `2099`
- `result_json`: `992`
- `result_csv`: `1015`
- `report_markdown`: `7`
- `log_file`: `10`
- `support_script`: `16`
- `database`: `2`
- `binary_artifact`: `2`
- `figure`: `1`
- `other`: `54`

Practical usage:

- use the JSON file for machine-readable auditing and downstream tooling,
- use the appendix for a human-readable inventory with category counts, high-signal artifacts, folder summaries, and explicit separation of branch-head vs working-tree artifacts.
- use the external benchmark matrix JSON as the compact, source-backed public SOTA target list for future benchmark scripts and literature refreshes.
- use the public SOTA targets JSON as the currently frozen benchmark threshold manifest for matched public comparisons.
- use the public SOTA gap analysis JSON to keep the external claim boundary explicit while public benchmark outputs are still missing.
- use the public on-target readiness JSON to track whether the canonical public on-target benchmark can actually be executed locally yet.
- use the remaining research browser prompt to close only the unresolved literature gaps without rerunning the whole external audit.

## What Has Been Done So Far

### Training And Model Development

Implemented and exercised in this branch:

- `scripts/train_on_target_trainval.py`
- `scripts/train_on_target_ranked.py`
- `scripts/train_on_target_deephf_style.py`
- `scripts/train_off_target_focal.py`
- `scripts/optuna_tune_on_target.py`
- `scripts/optuna_tune_off_target.py`
- `scripts/optimize_nonstacked_ensemble.py`
- OOF meta-stack / leaderboard update utilities
- conformal calibration and integrated design evaluation utilities

Architectures and search families already exercised:

- CNN-GRU style models
- cross-attention fusion variants
- gate fusion variants
- Mamba variants
- DeepHF-style variants
- DNABERT-inspired / pretrained encoder variants already present in branch workstreams
- non-stacked raw-prediction ensemble search
- OOF meta-stacking

### Optimization Modes Already Used

The branch has already used all of the following, not just one-off training:

- architecture sweeps
- seed sweeps
- neighborhood exploitation around the best recipe
- multi-cluster submission waves
- greedy and stochastic non-stacked blends
- stagewise blend searches
- tiny-alpha exact coordinate refinements
- Optuna TPE hyperparameter tuning for on-target
- Optuna TPE hyperparameter tuning for off-target

So the tuning story is real. It is not true that this branch has only done naive manual trial-and-error.

### Recent Non-Stacked On-Target Frontier Progression

Recent valid frontier progression:

- `w51` expanded valid best: `0.9108673834680832`
- `w51` expanded coordinate valid best: `0.9108779278031668`
- `w52` all-local coordinate valid best: `0.9108858595029408`
- `w52` pair-addition valid best: `0.9108880078772607`
- `w52` triple-addition valid best: `0.9108891526683431`
- `w53` full-pool pair valid best: `0.9108899744324618`
- `w55` full-pool coarse triple valid best: `0.9108902328051612`
- `w55b` exact tiny-alpha valid best: `0.9108903031450908`

Interpretation:

- The existing non-stacked prediction pool has been exploited very hard.
- Gains are now in the `1e-7` to `1e-4` regime.
- That means the present artifact pool is near saturation.
- The next meaningful gains need **new model diversity**, not more weight-search over the same predictions.

### Cluster And Infrastructure Work

Persistent reusable SSH sessions were established and validated for:

- `nibi` (GPU)
- `rorqual` (GPU)
- `fir` (GPU)
- `trillium` (CPU only)

Cluster viability status verified in this workstream:

- `nibi`: active, usable, GPU-capable
- `rorqual`: active, usable, GPU-capable
- `fir`: active, usable, GPU-capable
- `trillium`: active, CPU-only
- `cedar`: reachable but retired for compute; not usable
- `beluga`: reachable but compute service stopped; not usable
- `hulk`: unresolved hostname
- `graham`: unresolved hostname

Operational fixes already handled:

- re-established SSH after MFA gated new sessions,
- smoke-tested scripts before submitting jobs,
- corrected `fir` GPU account from `def-kwiese` to `def-kwiese_gpu`,
- hardened Optuna SQLite bootstrap against:
  - `alembic_version` races,
  - `UNIQUE constraint failed`,
  - `table ... already exists` races,
- aligned queued default seed ranges with actual Optuna search spaces.

## Latest External SOTA Research Snapshot (Primary Sources Checked)

This section is based on accessible primary sources checked on 2026-03-02. These sources define the current public benchmark landscape we must beat. They do **not** by themselves prove that we already beat it.

In the quick 2026 literature scan performed for this update, I did not find a newer clearly like-for-like benchmark paper for the same on-target/off-target prediction task that supersedes the 2025 benchmark papers below. As of this audit, the strongest directly comparable benchmark references I found remain the 2025 papers listed here.

### On-Target: Current Public Benchmark Leaders

Important update after a more detailed follow-on external full-text extraction:

- the current working external target should now be treated as `CRISPR_HNN` for the nine-dataset DeepHF-style single-model on-target benchmark,
- with `CRISPR-FMC` still remaining a strong and necessary comparator rather than the primary single-model leader.

#### 1) CRISPR-FMC (Frontiers in Genome Editing, 2025)

Source:

- [CRISPR-FMC: a dual-branch hybrid network for predicting CRISPR-Cas9 on-target activity](https://www.frontiersin.org/journals/genome-editing/articles/10.3389/fgeed.2025.1643888/full)

What is directly supported by the paper:

- Published on 2025-08-29.
- Evaluated across **nine public CRISPR-Cas9 datasets**.
- Claims to consistently outperform existing baselines in both Spearman and Pearson.

The paper explicitly lists the nine datasets and sample sizes:

- WT: `55,603`
- ESP: `58,616`
- HF: `56,887`
- xCas9: `37,738`
- SpCas9-NG: `30,585`
- Sniper-Cas9: `37,794`
- HCT116: `4,239`
- HELA: `8,101`
- HL60: `2,076`

Those numbers come directly from the article's dataset table.

Key concrete reported values in the paper:

- Large-scale dataset results:
  - WT SCC `0.861`, PCC `0.889`
  - ESP SCC `0.851`, PCC `0.845`
  - HF SCC `0.851`, PCC `0.866`
- Sniper-Cas9:
  - SCC `0.935`, PCC `0.957`
- HL60:
  - SCC `0.402`, PCC `0.404`
- Average across nine datasets in the ablation summary:
  - SCC `0.709`
  - PCC `0.716`
- Cross-dataset generalization to HL60:
  - train on WT -> test HL60: SCC `0.468`, PCC `0.459`
  - train on ESP -> test HL60: SCC `0.448`, PCC `0.432`

Why this matters:

- This is a concrete 2025 public benchmark target.
- If we want to claim we beat current on-target SOTA, we need like-for-like results on these public datasets and comparable protocols, not just a stronger score on our private Split A regime.

#### 2) CRISPR_HNN (Computational and Structural Biotechnology Journal, 2025)

Source:

- [Prediction of CRISPR-Cas9 on-target activity based on a hybrid neural network](https://pubmed.ncbi.nlm.nih.gov/40502933/)

What is directly supported by the abstract:

- Published 2025-05-27.
- Introduces `CRISPR_HNN`, combining MSC, MHSA, and BiGRU with one-hot and label encoding.
- The abstract states that it surpasses existing models on public datasets.

Why this matters:

- Even without relying on the full paper tables here, it is another 2025 on-target contender that must be considered in any honest SOTA comparison set.
- We should treat it as a reproducibility target until we benchmark against its public-dataset regime.

#### 3) CrnnCrispr (International Journal of Molecular Sciences, 2024)

Source:

- [CrnnCrispr: An Interpretable Deep Learning Method for CRISPR/Cas9 sgRNA On-Target Activity Prediction](https://www.mdpi.com/1422-0067/25/8/4429)

What is directly supported by the paper:

- Published 2024-04-17.
- Benchmarks on **nine public datasets** with varying sizes.
- Reports that CrnnCrispr outperforms four mainstream methods in accuracy and generalizability.
- Includes transfer learning for small datasets.
- Under leave-one-cell-out evaluation, the paper reports:
  - mean SCC `0.756` on large-scale datasets,
  - mean SCC `0.768` on medium-scale datasets,
  - strongest performance on large and medium regimes, with more room for improvement on small-scale data.
- For transfer learning on small datasets, it reports mean SCC values of:
  - HCT116: `0.768`
  - HELA: `0.709`
  - HL60: `0.545`
  after fine-tuning.

Why this matters:

- It remains a credible reproducible baseline family for the nine-dataset on-target benchmark regime that newer 2025 papers are still comparing against.

### Off-Target: Current Public Benchmark Leaders

#### 4) CCLMoff (Communications Biology, 2025)

Source:

- [A versatile CRISPR/Cas9 system off-target prediction tool using language model](https://www.nature.com/articles/s42003-025-08275-6)

What is directly supported by the paper:

- Published 2025-06-06.
- Introduces `CCLMoff`, a pretrained RNA language model framework.
- Built on a comprehensive dataset spanning **13 genome-wide deep sequencing techniques from 21 publications**.

The paper explicitly names the integrated detection technologies:

- Extru-seq
- SITE-seq
- CIRCLE-seq
- DISCOVER-seq
- DISCOVER-seq+
- CHANGE-seq
- BLESS
- GUIDE-seq
- Digenome-seq
- DIG-seq
- IDLV
- HTGTS
- SURRO-seq

Key concrete reported values in the paper:

- CIRCLE-seq cross-validation:
  - AUROC `0.985`
  - AUPRC `0.524`
- Train on CIRCLE-seq -> test on GUIDE-seq:
  - AUROC `0.996`
  - AUPRC `0.520`
- Compared benchmarks cited in the same cross-dataset setting:
  - CRISPR-IP: AUROC `0.945`, AUPRC `0.337`
  - MOFF: AUROC `0.876`, AUPRC `0.282`
  - CRISPR-DNT: AUROC `0.977`, AUPRC `0.381`
- Leave-one-dataset-out results reported in the paper:
  - DIG-seq: AUROC `0.985`, AUPRC `0.720`
  - DISCOVER-seq / DISCOVER-seq+: AUROC `0.944`, AUPRC about `0.665`
- The paper also notes CCLMoff can handle non-canonical sgRNA lengths and reports AUROC about `0.8123` on unseen 19/21 nt cases.

Why this matters:

- This is a stronger and more realistic external off-target benchmark regime than a single matched held-out split.
- Our internal off-target result is excellent, but it is not yet a like-for-like replacement for this broader cross-dataset benchmark suite.

#### 5) DNABERT-Epi (PLoS One, 2025)

Source:

- [Improved CRISPR/Cas9 off-target prediction with DNABERT and epigenetic features](https://pubmed.ncbi.nlm.nih.gov/41223195/)

What is directly supported by the abstract and figure descriptions:

- Published 2025-11-12.
- Benchmarked `DNABERT-Epi` against five state-of-the-art methods across **seven distinct off-target datasets**.
- The abstract states that the DNABERT-based models achieved competitive or superior performance.
- Figure descriptions show evaluation on real public datasets including:
  - Lazzarotto et al. (2020) GUIDE-seq
  - CHANGE-seq
  - TTISS
  - additional GUIDE-seq datasets from Listgarten et al. (2018), Chen et al. (2017), and Tsai et al.

Why this matters:

- This is another 2025 external competitor built around a foundation model plus real epigenetic context.
- It is relevant not because we must match its exact architecture, but because it raises the standard for what a current off-target benchmark suite should include.

### Source-Backed External Benchmark Matrix (Current Working Reference)

This is the compact benchmark matrix to use as the external target frame while building the public benchmark harness.

#### On-Target Reference Matrix

- `CRISPR_HNN` (2025)
  - task: on-target activity prediction
  - model type: single hybrid neural network
  - benchmark regime: nine DeepHF-style public datasets under matched five-fold CV
  - updated extracted values now being used as the frozen public target manifest:
    - WT SCC `0.877`
    - ESP SCC `0.861`
    - HF SCC `0.865`
    - xCas SCC `0.880`
    - SpCas9-NG SCC `0.866`
    - Sniper SCC `0.955`
    - HCT116 SCC `0.342`
    - HELA SCC `0.374`
    - HL60 SCC `0.427`
    - mean SCC `0.716`
  - direct implication:
    - this is now the primary single-model on-target target to beat under the matched nine-dataset public protocol

- `CRISPR-FMC` (2025)
  - task: on-target activity prediction
  - model type: single multimodal hybrid network
  - benchmark regime: nine public datasets plus cross-dataset transfer
  - strongest explicitly reported values in the paper:
    - WT SCC `0.861`, PCC `0.889`
    - ESP SCC `0.851`, PCC `0.845`
    - HF SCC `0.851`, PCC `0.866`
    - Sniper-Cas9 SCC `0.935`, PCC `0.957`
    - HL60 SCC `0.402`, PCC `0.404`
    - WT -> HL60 SCC `0.468`, PCC `0.459`
    - ESP -> HL60 SCC `0.448`, PCC `0.432`
  - direct implication:
    - these are concrete per-dataset and transfer targets we need to meet or exceed on the same public regime

- `CRISPR_HNN` (2025)
  - task: on-target activity prediction
  - model type: single hybrid neural network
  - benchmark regime: public on-target datasets
  - directly supported by source:
    - the abstract states it outperforms existing models on public datasets
  - direct implication:
    - treat as a current benchmark contender that must be reproduced and placed in our like-for-like comparison table

- `CrnnCrispr` (2024)
  - task: on-target activity prediction
  - model type: single interpretable deep learning model with transfer learning
  - benchmark regime: nine public datasets, including leave-one-cell-out and small-data transfer
  - explicitly reported values:
    - mean SCC `0.756` on large-scale datasets
    - mean SCC `0.768` on medium-scale datasets
    - transfer-learning SCC:
      - HCT116 `0.768`
      - HELA `0.709`
      - HL60 `0.545`
  - direct implication:
    - this remains a practical baseline floor for the nine-dataset public regime, especially on transfer

#### Off-Target Reference Matrix

- `CCLMoff` (2025)
  - task: off-target prediction
  - model type: single pretrained RNA language model system
  - benchmark regime: multi-assay, cross-dataset, leave-one-dataset-out
  - explicitly reported values:
    - CIRCLE-seq cross-validation: AUROC `0.985`, AUPRC `0.524`
    - CIRCLE-seq -> GUIDE-seq: AUROC `0.996`, AUPRC `0.520`
    - DIG-seq leave-one-dataset-out: AUROC `0.985`, AUPRC `0.720`
    - DISCOVER-seq / DISCOVER-seq+: AUROC `0.944`, AUPRC about `0.665`
    - unseen 19/21 nt cases: AUROC about `0.8123`
  - direct implication:
    - our internal off-target AUROC is strong, but we need comparable results on this broader public assay regime to claim external superiority

- `DNABERT-Epi` (2025)
  - task: off-target prediction
  - model type: single transformer foundation-model variant with epigenetic inputs
  - benchmark regime: seven real off-target datasets against five SOTA baselines
  - directly supported by source:
    - the abstract states competitive or superior performance across the seven datasets
  - direct implication:
    - this is a relevant public-dataset benchmark family for any 2025+ off-target SOTA claim

#### Integrated Design Benchmark Note

For integrated sgRNA design scoring, the literature is more fragmented than for pure on-target or off-target prediction:

- there is no single universally dominant public benchmark leaderboard comparable to the nine-dataset on-target regime or the multi-assay off-target regime,
- many tools combine on-target and off-target components under different utility definitions,
- and cross-paper comparisons often become invalid because the objective differs (editing efficacy, specificity, ranking utility, filtering heuristics, or end-to-end design score).

Practical conclusion:

- any honest integrated-design SOTA claim must define its utility metric explicitly,
- must benchmark against the strongest public on-target and off-target baselines used inside that utility,
- and should report utility metrics such as NDCG@k, Top-k hit rate, and ranking uplift over both component-only baselines.

### Cross-Cutting Benchmark Review

Source:

- [Benchmarking deep learning methods for predicting CRISPR/Cas9 sgRNA on- and off-target activities](https://academic.oup.com/bib/article/24/6/bbad333/7286387)
- [PubMed record](https://pubmed.ncbi.nlm.nih.gov/37775147/)

What is directly supported by the review:

- Published 2023.
- Benchmarks:
  - `10` mainstream deep learning on-target predictors on `9` public datasets
  - `8` representative off-target predictors on `12` public datasets
- The review's central conclusion is still operationally important:
  - most methods perform much better on larger datasets than on small-scale datasets,
  - many off-target methods look strong on balanced data but weaken on moderate and severe imbalance,
  - cross-dataset generalization is materially harder than within-dataset evaluation.

Why this still matters in 2026:

- This paper is not the newest, but it remains a strong benchmark design reference.
- It tells us how to avoid making an invalid SOTA claim by testing only on easy or favorable splits.

## What The External Literature Means For Our Claim

### What We Can Honestly Say Now

We can honestly say:

- The branch has achieved very strong internal performance.
- The branch has passed multiple proposal components.
- The branch is extremely close to the non-stacked on-target threshold.
- The branch's off-target matched held-out performance is very strong.

### What We Cannot Honestly Say Yet

We cannot yet honestly say:

- that we outperform **all latest SOTA** across current public benchmark suites,
- that we beat the 2025 on-target leaders on the full nine-dataset public regime,
- that we beat the 2025 off-target leaders on the broader cross-dataset public regime,
- or that the proposal is fully met under the branch's own non-stacked claim condition.

The reason is simple:

- Our best `0.9108903031` non-stacked score is on our **internal Split A protocol**.
- The latest public papers report performance on **different public datasets and protocols**.
- These are not interchangeable.

A stronger internal score on Split A does not automatically imply a stronger public benchmark result on WT/ESP/HF/xCas9/SpCas9-NG/Sniper/HCT116/HELA/HL60, or on GUIDE-seq/CHANGE-seq/CIRCLE-seq/DIG-seq/DISCOVER-seq suites.

## Why The Current Internal Result Is Still Not Enough For A Full SOTA Claim

There are four hard reasons.

### 1) Benchmark mismatch

Our main on-target claim frontier is on internal Split A.

Latest public on-target papers report on:

- nine public datasets,
- cross-dataset migration,
- leave-one-cell-out or five-fold cross-validation,
- and small-data transfer settings.

Until we run those same public regimes, we do not have a like-for-like comparison.

### 2) Non-stacked target still not passed

Even within our own proposal condition, the non-stacked target remains below threshold:

- required: `>= 0.911`
- current: `0.9108903031450908`

That is close, but not passable.

### 3) Off-target public benchmark breadth is larger than our current matched split

Our internal off-target result is very strong, but latest papers evaluate broader suites across multiple assay families and external validation regimes.

To claim external SOTA, we need to show performance across those real public datasets, not only our current matched held-out setup.

### 4) Not all cited public baselines are yet reproduced in this branch

The branch has strong implemented families, but a fully auditable "we beat the field" claim needs explicit, persisted like-for-like benchmark outputs against the currently relevant public comparators.

## Minimum Benchmark Program Required For A Defensible External SOTA Claim

If the goal is a real external SOTA claim, this is the minimum benchmark program.

### On-Target Benchmark Suite (Must Run)

Benchmark on the nine public datasets repeatedly used by recent papers:

- WT
- ESP
- HF
- xCas9
- SpCas9-NG
- Sniper-Cas9
- HCT116
- HELA
- HL60

At minimum, run all of the following protocols:

1. Independent within-dataset evaluation
   - match the paper protocol as closely as possible (five-fold CV where the paper uses five-fold, or the exact published split if available)

2. Leave-one-cell-out or leave-one-dataset-out transfer
   - especially large -> small transfer
   - explicitly include:
     - train on WT -> test HL60
     - train on ESP -> test HL60

3. Small-data transfer / fine-tuning regime
   - pretrain on large benchmark pool
   - fine-tune on HCT116 / HELA / HL60

Primary metrics:

- Spearman (SCC)
- Pearson (PCC)

What we need to beat:

- CRISPR-FMC's reported per-dataset and transfer results
- CRISPR_HNN's public-dataset claims
- CrnnCrispr's public-dataset and transfer-learning baselines

### Off-Target Benchmark Suite (Must Run)

Use real public assay-derived datasets, not synthetic-only negatives as the sole headline benchmark.

Core public datasets to include:

- GUIDE-seq
- CHANGE-seq
- CIRCLE-seq
- TTISS
- DIG-seq
- DISCOVER-seq
- DISCOVER-seq+

Expanded set strongly recommended:

- SITE-seq
- Digenome-seq
- Extru-seq
- IDLV
- HTGTS
- SURRO-seq
- BLESS

At minimum, run all of the following protocols:

1. Within-dataset cross-validation
2. Train on one assay, test on another assay
   - especially CIRCLE-seq -> GUIDE-seq
3. Leave-one-dataset-out across the combined assay pool
4. Variable-length sgRNA generalization where data permits

Primary metrics:

- AUROC
- AUPRC
- F1
- MCC
- balanced accuracy (where papers report it)

What we need to beat:

- CCLMoff on its external validation regime
- DNABERT-Epi on its real multi-dataset regime
- strong older comparators still cited in 2025 papers (CRISPR-Net, CRISPR-IP, MOFF, CRISPR-DNT)

### Claim Hygiene Rules

For any final "we beat SOTA" statement:

- do not mix internal Split A results with public benchmark results,
- do not mix stacked/OOF and non-stacked unless explicitly labeled,
- do not count surrogate-objective wins that fail true-metric back-check,
- do not use a single favorable dataset as the whole story,
- and do not claim cross-paper superiority without matching the protocol closely enough to be fair.

## How To Improve The Single Non-Stacked Model (Not Just Ensembles)

The branch has already squeezed the existing non-stacked prediction pool hard. If the goal is to win with a single unstacked model, the highest-yield work is not more blend tuning. It is better base models and better benchmark alignment.

### Most Likely High-Yield Technical Moves

#### 1) Shift the optimization target toward public benchmark alignment

Right now, the branch's hardest remaining blocker is an internal Split A threshold. But if we also want public SOTA, the single-model work must optimize for:

- multi-dataset robustness,
- cross-cell transfer,
- and low-data adaptation.

That means the single-model objective should not be "best on one split only".

#### 2) Use one model trained across the full public on-target pool

For a serious single-model push, use a single shared backbone trained across the nine on-target public datasets with:

- dataset or cell-line tokens,
- shared trunk,
- correlation-aware regression head,
- and optional per-dataset calibration heads only if needed for stability.

This should improve robustness compared with training separate models and later blending them.

#### 3) Train with a loss that better tracks the ranking target

Since the headline metric is Spearman, a pure regression loss is suboptimal.

Recommended direction:

- hybrid loss = robust regression loss + differentiable rank / pairwise ordering component

Example practical combination:

- Huber or MAE component for amplitude stability
- plus pairwise ranking loss or differentiable correlation surrogate for order consistency

This is a direct path to improving single-model Spearman without depending on ensembles.

#### 4) Use stronger multimodal priors, but keep the final model single

The recent public winners point in the same direction:

- pretrained representations help,
- multimodal fusion helps,
- cross-attention or explicit interaction modeling helps,
- but those gains can still live inside a single deployed model.

The practical target for a next single-model winner is:

- one-hot branch for direct positional specificity,
- pretrained nucleotide or RNA/DNA embedding branch,
- explicit interaction module (cross-attention or gated bilinear fusion),
- one final scalar head.

That remains a single model, not a stack.

#### 5) Spend compute on model diversity, not more reweighting of old predictions

This is the main current engineering conclusion.

Because the non-stacked blend frontier is already near-saturated, the next `1e-4` will more likely come from:

- fresh high-quality single-model checkpoints,
- new training curricula,
- better loss alignment,
- or better public-data pretraining,

not from another round of tiny blend-coordinate search over the same files.

### Practical Single-Model Experiment Plan

The most defensible next single-model program is:

1. Build one public-benchmark-aligned backbone
   - keep the strongest existing branch trunk (cross-attention / gated multimodal encoder)
   - keep it as one deployed model

2. Train on merged public on-target datasets
   - strict dedupe by guide sequence and PAM
   - explicit prevention of train/test leakage across duplicated guides or near-duplicates

3. Use Optuna or ASHA for the model itself, not just for local neighborhood sweeps
   - tune learning rate, width, dropout, batch size, warmup, weight decay, correlation-loss weight, pretrain fraction, and seed

4. Evaluate first on public regimes, then on internal Split A
   - if the model cannot hold up on public regimes, it is not a real SOTA candidate even if Split A improves

## Data Quality Standard: Real, Verified, Benchmarkable Datasets Only

The requirement to outperform on real, reliable datasets is correct. That means the benchmark program should prefer:

- public assay-derived datasets used repeatedly in the literature,
- datasets with explicit provenance and cited source publications,
- external validation across assay families,
- and fixed benchmark manifests checked into the repo.

It should avoid using as the main headline benchmark:

- synthetic-only constructed evaluation sets,
- toy subsets,
- or one-off private splits without public analogs.

This does not mean synthetic negatives are never useful.

It means:

- they can be part of training or auxiliary evaluation,
- but they should not be the only basis for a strong external SOTA claim.

## Concrete Repository Deliverables Needed Next

To turn this into a truly defensible benchmark package, the repo should next produce these persisted artifacts.

### 1) Fixed public benchmark manifests

Create versioned manifests for:

- on-target nine-dataset suite
- off-target multi-assay suite

Each manifest should record:

- source paper
- raw source file checksum
- filtering rules
- dedupe rules
- exact split protocol

### 2) Like-for-like benchmark runner outputs

For each baseline and for our best single model, persist:

- prediction files
- per-dataset metrics JSON
- summary tables
- exact training config
- seed
- code revision

### 3) One claim table with strict labels

A final claim table should have separate rows for:

- internal Split A non-stacked
- internal Split A stacked/OOF
- strict Split A
- public on-target nine-dataset benchmark
- public off-target multi-assay benchmark

Without that separation, the final story will be too easy to misstate.

## Recommended Immediate Next Actions

This is the shortest path to a defensible result.

1. Finish the public benchmark harness
   - build and freeze the public on-target and off-target manifests
   - make the benchmark protocol explicit and reproducible

2. Run the best current single model on the public on-target nine-dataset suite
   - this tells us immediately whether the current family is actually SOTA-competitive outside Split A

3. Reproduce at least the strongest 2025 comparators as benchmark references
   - CRISPR-FMC-style regime for on-target
   - CCLMoff-style regime for off-target

4. Launch a new single-model training wave aimed at robustness, not just local Split A score
   - one strong multimodal single model
   - correlation-aware objective
   - public-data-aligned training curriculum

5. Only after that, resume aggressive non-stacked ensemble search on the expanded pool
   - because fresh model diversity is now the missing ingredient

## Public Benchmark Freeze (Current External Targets)

The public benchmark targets are now frozen locally in:

- `PUBLIC_SOTA_TARGETS_2026-03-02.json`
- `PUBLIC_ON_TARGET_READINESS_2026-03-02.json`

This file should be treated as the current benchmark threshold manifest for the next execution phase.

Current frozen public target leaders:

- on-target public leader status: dual-threshold split leadership (`CRISPR_HNN` for the nine-dataset average target, `CRISPR-FMC` for per-dataset / transfer / small-data targets)
- off-target single-model public leader: `CCLMoff`

The first structured local claim-hygiene comparison against those targets is recorded in:

- `PUBLIC_SOTA_GAP_ANALYSIS_2026-03-02.json`

Its main result is:

- external public SOTA claim is still **not valid now**, not because the local models are necessarily weaker, but because matched public benchmark outputs are still missing in the current local artifact set.

Execution rule going forward:

- run the public on-target benchmark in both modes:
  - separate-dataset evaluation for each canonical public dataset
  - pooled summary evaluation after all separate-dataset outputs exist
- after the canonical suite is complete, run additional public datasets as secondary evidence using the same separate-then-pooled pattern, but do not substitute them for the canonical claim benchmark

This is now the required execution style for a defensible public on-target comparison.

The supporting comparison helper is:

- `scripts/compare_against_public_sota.py`

This gives us a clean mechanism to keep the external-claim boundary explicit as public benchmark results are added.

## AI Researcher Prompt (For A Full External Benchmark Audit)

Use the following prompt verbatim or with only minimal adaptation when asking an AI researcher or research assistant model to perform the external benchmark and SOTA audit.

```text
You are acting as a rigorous scientific benchmarking researcher for a CRISPR guide design and prediction project. Your task is to produce a complete, evidence-backed external audit of the latest state of the art (SOTA) for CRISPR on-target activity prediction, off-target prediction, and integrated sgRNA design ranking.

Your goals are:

1. Identify the latest and strongest publicly reported SOTA models, benchmark papers, and reproducible baselines for:
   - on-target activity prediction
   - off-target prediction
   - integrated sgRNA ranking / design scoring
   - calibration / uncertainty if reported

2. Build a benchmark inventory of the real, public, assay-derived datasets used across the literature.
   Do not focus on synthetic-only or toy datasets unless they are explicitly standard and widely used as auxiliary benchmarks.

3. For every important model and every important dataset, extract the best reported performance values and the exact evaluation protocol.

4. Separate clearly between:
   - within-dataset results
   - cross-dataset transfer results
   - leave-one-dataset-out or leave-one-cell-out results
   - fine-tuning / transfer-learning results
   - stacked / ensemble methods
   - single non-stacked models

5. Produce a final recommendation on what benchmark suite and what target values a new model must beat to make an honest “we outperform current SOTA” claim.

Required scope:

- Cover papers through the latest available date, prioritizing 2024, 2025, and 2026 if available.
- Prefer primary sources only: journal papers, conference papers, official supplementary materials, official documentation, and official benchmark repositories.
- Include source links for every claim.
- If a paper is inaccessible, say exactly what is unavailable and do not invent numbers.

Deliverables:

A. Executive summary
- What are the current strongest known models for on-target, off-target, and integrated design?
- What are the current strongest single non-stacked models?
- What is the latest benchmark standard that a serious new model must be tested on?

B. Complete benchmark table
For each paper/model, provide:
- model name
- year
- task (on-target / off-target / integrated design / calibration)
- architecture summary
- whether it is a single model or ensemble/stacked method
- datasets used
- sample sizes if available
- evaluation protocol (CV, hold-out, transfer, leave-one-out, etc.)
- primary metrics (Spearman, Pearson, AUROC, AUPRC, F1, MCC, etc.)
- best reported values
- exact context of those values (which dataset, which split, which transfer direction)
- strengths
- weaknesses / caveats
- source URL

C. Dataset audit
Produce a complete dataset inventory of the real benchmark datasets used across papers.
For each dataset, provide:
- dataset name
- task type
- source publication
- assay or measurement type
- size
- class balance if relevant
- common preprocessing or filtering choices
- common benchmark protocols
- which major papers use it
- whether it is considered a primary benchmark or only auxiliary

D. SOTA-by-dataset table
For each major dataset, identify:
- the best known reported model(s)
- the best single-model result
- the best ensemble/stacked result if different
- the exact metric values
- whether the comparison is fair and like-for-like

E. Claim-validity framework
Define the minimum conditions required to honestly claim:
- “better than current on-target SOTA”
- “better than current off-target SOTA”
- “better than current integrated sgRNA design SOTA”
- “better as a single non-stacked model”

This section must explicitly state what would NOT be a valid claim (for example, comparing a private split to a public benchmark without protocol matching).

F. Recommended benchmark suite for our project
Propose the exact benchmark suite we should run if our goal is to make a defensible SOTA claim.
This must include:
- the exact public datasets to include
- the exact evaluation protocols to include
- the exact metrics to report
- which baselines must be reproduced
- which comparisons must be labeled as single-model vs ensemble

Strict methodology requirements:

- Do not rely on one paper alone.
- Cross-check claims across multiple sources where possible.
- Prefer exact quoted numeric results from primary sources, but keep quotes short and compliant.
- If two papers report different values for “best” on the same dataset, explain why (different split, different preprocessing, different metric, different data leakage controls, etc.).
- Explicitly flag when a result is not directly comparable.
- Explicitly flag when a benchmark uses synthetic negatives, synthetic augmentation, or private splits.
- Explicitly flag when a model depends on extra modalities (epigenetics, assay metadata, pretrained language models, etc.).

Output format:

1. Executive Summary
2. Latest SOTA Models
3. Dataset Inventory
4. SOTA By Dataset
5. Single-Model vs Ensemble Leaders
6. Claim-Validity Rules
7. Recommended Benchmark Program
8. Source List

The final output must be detailed, conservative, source-backed, and suitable for use in a thesis or publication audit. If a fact is uncertain, say so directly instead of guessing.
```

## Bottom Line

The branch is strong, but the claim must stay precise.

Current truthful status is:

- We have **not** yet fully satisfied the non-stacked on-target proposal threshold.
- We have **not** yet established a fair, public-dataset, like-for-like demonstration that we outperform all latest external SOTA models.
- We **do** have a credible path to get there, but it requires public benchmark execution, not just internal split optimization.

The strongest next move is therefore:

- stop treating the remaining job as only a local Split A tuning problem,
- and convert it into a full public benchmark reproduction and single-model robustness program.

That is the standard required for a defensible "we beat SOTA" result.

## Execution Update (2026-03-02)

Execution work has now started and the repo contains runnable acquisition and readiness artifacts:

- `on_target_dataset_manifest.csv`
- `off_target_dataset_manifest.csv`
- `dataset_download_links.json`
- `dataset_reconstruction_requirements.json`
- `canonical_benchmark_acquisition_checklist.json`
- `offtarget_negative_construction_audit.csv`
- `benchmark_protocol_matrix.csv`
- `public_claim_thresholds.json`

The local acquisition/staging workflow is now:

- `scripts/acquire_public_benchmarks.py`
- `scripts/prepare_public_benchmark_inputs.py`
- `scripts/evaluate_public_benchmark_readiness.py`
- `scripts/run_public_benchmark_harness.py`
- wrapper: `scripts/acquire_public_benchmarks.sh`
- env bootstrap: `scripts/bootstrap_public_benchmark_env.sh`

Current state after running those steps:

- the full canonical nine-dataset on-target suite is staged locally under `data/public_benchmarks/on_target/canonical_9`
- the staged CSVs have the expected `sgRNA,indel` schema
- the on-target public benchmark is now ready for matched evaluation
- the modern primary off-target suite is still incomplete because the CCLMoff compiled bundle and some supplement-derived tables are not yet staged locally
- the legacy secondary off-target fast-start bundle has partial local support via cloned repos and staged mirror files

Hard blocker for actually running the heavy benchmark training/evaluation scripts on this machine right now:

- the active local `python3` environment is missing `numpy`, `pandas`, and `torch`

So the benchmark plumbing is now in place, but actual claim-valid model runs still require:

- a Python environment with the ML dependencies installed
- staging the remaining modern primary off-target data
- then running single-model public benchmarks before any ensemble public benchmarks

That Python environment blocker has now been cleared with:

- `.venv-public-benchmark`
- `requirements-public-benchmark.txt`
- `scripts/bootstrap_public_benchmark_env.sh`

Public on-target execution is no longer theoretical:

- deterministic public folds are now generated by `scripts/build_public_on_target_folds.py`
- transfer splits are generated by `scripts/build_public_on_target_transfer_split.py`
- the full benchmark wrapper is `scripts/run_public_on_target_benchmark.py`
- a 1-fold / 1-epoch smoke aggregation is saved at `results/public_benchmarks/smoke/SMOKE_SUMMARY_2026-03-02.json`

Important interpretation:

- the smoke runs are only operational validation
- they are explicitly **not** claim-valid benchmark results
- they prove the public on-target path is runnable end-to-end on this repo

Latest execution update:

- a real 5-fold public on-target sweep is now running against the canonical suite
- `WT`, `ESP`, and `HF` have completed valid 5-fold single-model results under the public harness
- an input-scaling bug was detected in the staged `xCas9`, `SpCas9-NG`, and `Sniper-Cas9` datasets:
  - those three had been copied with raw percentage-like `indel` labels instead of normalized `[0,1]` labels
  - the staging pipeline now conditionally min-max normalizes only datasets whose `indel` column is outside `[0,1]`
  - the affected fold files were rebuilt and the invalid partial results were discarded
- after the normalization fix, the rerun is continuing from `xCas9` onward in the active public benchmark session
- the current interim multi-fold summary is stored at:
  - `results/public_benchmarks/full_runs/INTERIM_FULL_SUMMARY_2026-03-02.json`
- the current execution state is tracked in:
  - `PUBLIC_EXECUTION_STATUS_2026-03-02.json`

Current reality:

- the on-target public benchmark path is now fully operational and producing real matched results
- those results are still materially below the frozen external thresholds so far
- the off-target primary CCLMoff bundle is staged locally, but the DNABERT-Epi supplement tables and processed CHANGE-seq secondary artifacts are still incomplete

Current matched public on-target result for the baseline configuration (`cnn_gru`, `d_model=16`, `mse`, 3 epochs):

- full 9-dataset 5-fold CV is complete
- matched `WT -> HL60` transfer is complete
- consolidated result artifact:
  - `results/public_benchmarks/FULL_PUBLIC_ON_TARGET_SUMMARY_2026-03-02.json`

Current baseline outcome:

- 9-dataset mean SCC: `0.3037843526`
- gap vs frozen CRISPR_HNN mean target `0.716`: `-0.4122156474`
- `WT` mean SCC: `0.6537818222` vs target `0.861`
- `ESP` mean SCC: `0.6355124184` vs target `0.851`
- `HF` mean SCC: `0.6305552316` vs target `0.865`
- `Sniper-Cas9` mean SCC: `0.2254583575` vs target `0.935`
- `HL60` mean SCC: `-0.0393834085` vs target `0.402`
- `WT -> HL60` transfer mean SCC: `0.2153413840` vs target `0.468`

So the public single-model on-target benchmark is now complete for this baseline, and it is clearly **not** yet SOTA-competitive under the frozen external thresholds.
