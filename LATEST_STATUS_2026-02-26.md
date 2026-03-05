# Latest Status (2026-02-27)

Timestamp (UTC): `2026-03-02T19:00:00Z`

## Direct answer
- Have we for-real fully met all proposal targets under the defined conditions? **No (not yet).**
- Have we claim-valid outperformed latest external SOTA across all aspects? **No (not yet claimable).**

## Current best metrics
- Split A (stacked/OOF pool):
  - Best single: `0.9700928819`
  - Best ensemble: `0.9702816198`
- Split A (non-stacked claim-valid pool):
  - Best single: `0.8892076527` (`results/runs/w50opt_nibi_w6_t0065.json`)
  - Best ensemble: `0.9108903031` (`results/runs/w55b_exact_tiny_alpha_full90_summary.json`)
  - Gap to `0.911`: `0.0001096969`
  - Delta vs baseline `0.876`: `+0.0348903031` (still short of the `>= 0.035` target by `0.0001096969`)
  - Important validity note: a rank-space surrogate optimizer produced `0.9112039570`, but when back-checked under the true Spearman metric it dropped to `0.9105177498`; that surrogate result is **not counted**.
- Split A strict regime (best single): `0.4024609009`
- Off-target matched held-out test (pair-aware):
  - AUROC: `0.9999832097`
  - AUPRC: `0.9999999194`
- Conformal coverage: `0.9034172662` (target `0.90 +/- 0.02`)

## Proposal-condition pass/fail snapshot
- On-target non-stacked single >= 0.911: **FAIL**
- On-target non-stacked ensemble >= 0.911: **FAIL**
- On-target delta vs 0.876 >= 0.035 (non-stacked ensemble): **FAIL**
- On-target stacked/OOF >= 0.911: **PASS**
- Strict Split-A >= 0.911: **FAIL**
- Off-target pair-aware AUROC >= 0.92: **PASS**
- Off-target pair-aware AUROC >= 0.99: **PASS**
- Off-target pair-aware improvement significance (`p < 0.001`): **PASS**
- Conformal coverage within `0.90 +/- 0.02`: **PASS**
- Integrated score beats on/off baselines on NDCG@20 and Top20 utility: **PASS**
- On-target significance vs baseline (`p < 0.001`): **PASS**

Overall `proposal_targets_all_met_nonstacked_claim`: **false**

## New artifacts added in this continuation
- `results/runs/w50opt_nibi_w2_summary.json`
- `results/runs/w50opt_nibi_w4_summary.json`
- `results/runs/w50opt_nibi_w5_summary.json`
- `results/runs/w50opt_nibi_w6_summary.json`
- `results/runs/w50opt_nibi_w6_t0065.json`
- `results/runs/w50opt_nibi_w6_t0065_predictions.csv`
- `results/runs/w51opt_offtarget_nibi_submitted_jobs_20260227_221917.txt`
- `results/runs/w51opt_offtarget_rorqual_submitted_jobs_20260227_221917.txt`
- `results/runs/w51opt_offtarget_rorqual_resubmitted_20260227_223823.txt`
- `results/runs/w51_runtime_snapshot.json`
- `results/runs/w51_optuna_stochastic_search_summary.json`
- `results/runs/w51_optuna_stochastic_search_predictions.csv`
- `results/runs/w51_optuna_expanded_search_summary.json`
- `results/runs/w51_optuna_expanded_search_predictions.csv`
- `results/runs/w51_optuna_expanded_coordinate_summary.json`
- `results/runs/w51_optuna_expanded_coordinate_predictions.csv`
- `results/runs/w52_all_local_optuna_coordinate_summary.json`
- `results/runs/w52_all_local_optuna_coordinate_predictions.csv`
- `results/runs/w52_pair_addition_search_summary.json`
- `results/runs/w52_pair_addition_search_predictions.csv`
- `results/runs/w52_triple_addition_search_summary.json`
- `results/runs/w52_triple_addition_search_predictions.csv`
- `results/runs/w53_fullpool_pair_search_summary.json`
- `results/runs/w53_fullpool_pair_search_predictions.csv`
- `results/runs/w54_local_stochastic_around_fullpool_pair_summary.json`
- `results/runs/w54b_exact_tiny_alpha_full90_summary.json`
- `results/runs/w55_fullpool_triple_search_summary.json`
- `results/runs/w55_fullpool_triple_search_predictions.csv`
- `results/runs/w55b_exact_tiny_alpha_full90_summary.json`
- `results/runs/w55b_exact_tiny_alpha_full90_predictions.csv`
- `results/runs/w58pilot_fir_submitted_jobs_20260302_190000.txt`
- `results/runs/w59opt_nibi_submitted_jobs_20260302_185705.txt`
- `results/runs/w59opt_rorqual_submitted_jobs_20260302_185705.txt`
- `results/runs/w51opt_offtarget_nibi_w3_summary.json`
- `results/runs/w51opt_offtarget_rorqual_w1_summary.json`
- `results/runs/w51opt_offtarget_rorqual_w2r_summary.json`
- `results/runs/w51opt_offtarget_rorqual_w3_summary.json`
- `results/runs/w51opt_offtarget_rorqual_w4_summary.json`
- `results/runs/w51opt_offtarget_nibi_resubmitted_20260228_020000.txt`
- `results/runs/w51_optuna_rankspace_search_summary.json` (surrogate-only; not claim-valid)
- `results/runs/w51_optuna_rankspace_search_predictions.csv` (surrogate-only; not claim-valid)
- `scripts/optuna_tune_on_target.py`
- `scripts/optuna_tune_off_target.py`

## Current runtime state
- `w50opt` (on-target Optuna):
  - `nibi`: completed trials seen locally `72`; best trial `0.8892076527`
  - `rorqual`: completed trials seen remotely `18`; best trial `0.8849529368`
- `w51` on-target non-stacked refinement:
  - best valid progression this turn:
    - two-model blend: `0.9054696361`
    - greedy top-10 blend: `0.9104242534`
    - stochastic top-10 blend: `0.9106131811`
    - expanded valid best: `0.9108673835`
    - expanded coordinate-refined valid best: `0.9108779278`
    - all-local Optuna coordinate valid best: `0.9108858595`
    - pair-addition valid best: `0.9108880079`
    - triple-addition valid best: `0.9108891527`
    - full-pool pair valid best (all `90` local `w50opt` predictions): `0.9108899744`
    - full-pool coarse triple valid best: `0.9108902328`
    - full-pool exact tiny-alpha valid best: `0.9108903031`
  - `w53` tight local stochastic refinement around the triple best found no further gain
  - `w54` local stochastic refinement around the full-pool pair best found no further gain
  - `w54b` exact tiny-alpha full-90 coordinate sweep found no further gain
  - `w55b` did find two additional `1e-05` exact tiny-alpha improvements, but the frontier is still a near-miss
  - full `w50opt` local pool is now present (`90` prediction files)
  - surrogate rank-space pass exceeded `0.911`, but failed true-metric back-check and is not counted
- `w51` off-target Optuna:
  - `nibi`: one worker completed at `best_auroc = 1.0`, `best_auprc = 1.0`; three initial workers failed on a schema-creation race and were resubmitted as `9465324`, `9465325`, `9465326`
  - `rorqual`: all four workers completed; best summaries report `best_auroc = 1.0`, `best_auprc = 1.0`
  - retry logic now covers both the `alembic_version`/unique-constraint race and the `table ... already exists` schema race
  - all resubmitted `nibi` workers now completed successfully at `best_auroc = 1.0`, `best_auprc = 1.0`
  - additional off-target GPU tuning is intentionally paused because the current tuned runs are already saturated at `1.0`

## Cluster execution status
- Persistent SSH sessions are now live and reusable for:
  - `nibi`
  - `rorqual`
  - `fir`
  - `trillium` (CPU-only)
- Additional cluster viability checks:
  - `cedar`: reachable, but retired for compute; do not submit jobs there
  - `beluga`: reachable, but compute service stopped; do not submit jobs there
  - `hulk`: hostname unresolved
  - `graham`: hostname unresolved
- `fir` was smoke-tested successfully after syncing the scripts and correcting the GPU account from `def-kwiese` to `def-kwiese_gpu`
- Current newly submitted jobs:
  - `fir` pilots:
    - on-target: `25304365`
    - off-target: `25304366`
    - both currently pending
  - fresh on-target Optuna wave:
    - `nibi`: `9665804`, `9665805`
    - `rorqual`: `7560367`, `7560368`
    - all currently pending
- Local long-running refinement searches were also launched; no additional claim-valid uplift from them has been recorded at this snapshot.

## Full step-by-step execution log
- See: `RUN_LOG_2026-02-25_2026-02-26.md`
