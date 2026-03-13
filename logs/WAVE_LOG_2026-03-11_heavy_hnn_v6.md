# Heavy Wave Log — 2026-03-11 15:25 PDT

## Context
- **Cluster:** rorqual (Alliance Canada)
- **Wave type:** Protocol-aligned CRISPR_HNN v6 retrain + WT→HL60 transfer
- **Submission method:** Inline env vars via `sbatch` (not `--export`)
- **Wrapper fix applied:** REPO_DIR and VENV_DIR defaults changed from `$HOME` to `/scratch/amird/` (commit `b733a2f`)

## Root Cause Fix
Smoke tests 8052855 and 8053272 failed because SLURM wrappers defaulted to
`$HOME/chromaguide_experiments` which contained a stale copy of the code (Mar 5, 7851 bytes)
without `--lr` and `--patience` argparse entries. The correct code lives at
`/scratch/amird/chromaguide_experiments` (Mar 11, 8176 bytes). Fixed in commit `b733a2f`.

## Corrected Smoke Test
- **Job:** 8053659 on rorqual
- **Config:** WT, 1 fold, 5 epochs, batch_size=16, lr=1e-4, patience=3, max_rows=500, seed=2024
- **Result:** PASSED — test_scc=0.2417 (expected low for 500-row 5-epoch smoke)
- **Confirmed:** Ran from `/scratch/amird/` with correct env and accepted all args

## Invalid Submissions Cancelled
- Jobs 8053762–8053773 (12 jobs): Cancelled because `--export=ALL,DATASETS=WT.ESP.HF`
  passed dot-separated datasets instead of comma-separated. Inline env vars used instead.

## Protocol
| Parameter | Value |
|-----------|-------|
| epochs | 200 |
| batch_size | 16 |
| lr | 1e-4 |
| patience | 10 |
| folds | 5 (default) |
| max_rows | 0 (all data) |
| time_limit | 24:00:00 |
| GPU | 1×GPU (SLURM default) |
| Memory | 64G |
| CPUs | 8 |

## Dataset Group Rationale
Split into 3 groups to limit per-job runtime and isolate failures:
- **Group A (DeepHF anchor):** WT, ESP, HF — the canonical benchmark trio
- **Group B (variant Cas9):** xCas9, SpCas9-NG, Sniper-Cas9 — enzyme variants
- **Group C (cell-line pressure):** HCT116, HELA, HL60 — cell-line transfer targets

## Canonical HNN Jobs (9)
| Job ID | Seed | Group | Datasets | Run Tag |
|--------|------|-------|----------|---------|
| 8053822 | 2024 | A | WT,ESP,HF | sota_hnn_v6_groupA_e200_s2024 |
| 8053823 | 2024 | B | xCas9,SpCas9-NG,Sniper-Cas9 | sota_hnn_v6_groupB_e200_s2024 |
| 8053824 | 2024 | C | HCT116,HELA,HL60 | sota_hnn_v6_groupC_e200_s2024 |
| 8053825 | 220 | A | WT,ESP,HF | sota_hnn_v6_groupA_e200_s220 |
| 8053826 | 220 | B | xCas9,SpCas9-NG,Sniper-Cas9 | sota_hnn_v6_groupB_e200_s220 |
| 8053827 | 220 | C | HCT116,HELA,HL60 | sota_hnn_v6_groupC_e200_s220 |
| 8053828 | 42 | A | WT,ESP,HF | sota_hnn_v6_groupA_e200_s42 |
| 8053829 | 42 | B | xCas9,SpCas9-NG,Sniper-Cas9 | sota_hnn_v6_groupB_e200_s42 |
| 8053830 | 42 | C | HCT116,HELA,HL60 | sota_hnn_v6_groupC_e200_s42 |

## Transfer Jobs (3)
| Job ID | Seed | Source | Target | Run Tag |
|--------|------|--------|--------|---------|
| 8053831 | 2024 | WT | HL60 | sota_hnn_transfer_WT_HL60_v6_e200_s2024 |
| 8053833 | 220 | WT | HL60 | sota_hnn_transfer_WT_HL60_v6_e200_s220 |
| 8053834 | 42 | WT | HL60 | sota_hnn_transfer_WT_HL60_v6_e200_s42 |

## Execution Notes
- All heavy jobs are cluster-only (rorqual)
- No local heavy computation
- HPO tuning wave deferred until this baseline completes
- Multi-cluster expansion (fir, nibi, narval) in progress for future waves
- Thesis-grade claim discipline maintained throughout

## Standing Hard Truths
- CCLMoff primary off-target: blocked without provenance
- crispAI uncertainty: frozen per UNCERTAINTY_CLAIM_FREEZE_2026-03-11.md
- DeepHF: canonical WT/ESP/HF anchor
- CRISPR_HNN / CRISPR-FMC: retrain-capable repos, not frozen public checkpoint repos
- WT→HL60 and HL60: key on-target pressure points


## 2026-03-12 Outcome Snapshot
- `squeue` on rorqual is now empty for the v6 wave.
- **Group C completed on all 3 seeds**:
  - `8053824` seed 2024: mean_scc_across_datasets `0.3052578624`
  - `8053827` seed 220: mean_scc_across_datasets `0.3034365108`
  - `8053830` seed 42: mean_scc_across_datasets `0.2992228551`
- These runs produced HCT116/HELA/HL60 summaries but **did not move the strict board**; best standing HL60 remains above them.
- **Group A** only emitted `WT_fold0.json` for all three seeds and never produced `SUMMARY.json`.
- **Group B** emitted partial `xCas9` folds only; jobs `8053823` and `8053826` were explicitly cancelled by Slurm due to node failure.
- **Transfer jobs** `8053831`, `8053833`, `8053834` emitted no result artifacts despite log startup; transfer directories exist but are empty.
- Multiple jobs logged `CUDA_ERROR_NO_DEVICE`, so the next step is failure isolation and repaired reruns, not blind duplication.

## 2026-03-12 Nibi Repaired Rerun Wave
- Promotion smoke `10196496` completed with real `SUMMARY.json`; wrapper logged `TENSORFLOW_GPU_COUNT=1`.
- Repaired reruns moved only the failed or missing `rorqual` v6 components to `nibi`: Group A, Group B, and WT→HL60 transfer.
- Completed jobs: `10196523`–`10196531`.
- Best repaired outcomes:
  - `WT`: `0.8472109768` (new strict best, still below `0.861`)
  - `HF`: `0.8398865386` (new strict best, still below `0.865`)
  - `Sniper-Cas9`: `0.9298523422` (new strict best, still below `0.935`)
  - `WT→HL60`: `0.4630754299` (below standing best `0.4653505492`)
- No frozen threshold flipped in this rerun wave.
