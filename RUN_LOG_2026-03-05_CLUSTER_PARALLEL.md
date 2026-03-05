# Cluster Parallel Run Log (2026-03-05)

## Scope
This log captures the cluster-wide smoke validation, parallel full submissions, Optuna waves, and current metric status for public claim benchmarking.

## Smoke Validation (per cluster)
- nibi: PASS (job `9838069`, completed)
- rorqual: PASS (job `7727809`, completed)
- fir: PASS by execution evidence (multiple full jobs running successfully on `fc11005`)
- cedar: PASS (job `66521807`, output `SMOKE_OK` at `/scratch/amird/smoke/codex_smoke_66521807.out`)
- trillium: PASS (job `1126184`, completed)
- beluga: BLOCKED (Slurm plugin incompatibility: `cc-tmpfs_mounts.so`)

## Parallel Submissions (today)

### nibi
- Running
  - `9836031` `pub_on_full`
  - `9838859` `pub_on_optuna`
  - `9839196` `pub_off_sweep` (new all-split manifest sweep)
- Completed
  - `9838669` `pub_off_cclmoff`
  - `9838862` `pub_off_optuna`
  - `9836395` `sota_fmc_pub`

### rorqual
- Running
  - `7728103` `pub_on_full`
- Pending
  - `7728224` `pub_on_optuna`
  - `7728225` `pub_off_optuna`
  - `7728813` `pub_off_sweep` (new all-split manifest sweep)
- Completed
  - `7728104` `pub_off_cclmoff`

### fir
- Running
  - `25783647` `pub_on_full`
  - `25788418` `pub_on_optuna`
  - `25789949` `pub_off_sweep` (new all-split manifest sweep)
  - `25786572` `sota_fmc_pub`

### cedar
- Pending (node availability constrained)
  - `66521805` `pub_on_full`
  - `66521806` `pub_off_cclmoff`
  - `66521935` `pub_on_optuna`
  - `66521936` `pub_off_optuna`
  - `66522228` `pub_off_sweep`

### trillium
- Running
  - `1119483` `pub_on_opt_cpu`
  - `1119950` `pub_on_opt_cpu`
  - `1124392` `pub_on_opt_cpu` (full)
  - `1124393` `pub_off_opt_cp` (full)

## Off-target Pipeline Correction Added
Problem found:
- Prior manifest runs using `train_public_off_target_cclmoff.py` with LODO manifest were evaluating a single split (`fold_index=0`) per job, not aggregated all held-out methods.

Fix added:
- New sweep runner: `scripts/run_public_off_target_manifest_sweep.py`
- New SLURM wrapper: `scripts/slurm_public_off_target_manifest_sweep.sh`
- These execute every manifest split and emit one aggregate summary file.

## Best Completed Metrics Snapshot (as of log time)

### On-target (best completed across clusters)
- Best mean 9-dataset SCC: `0.5522381826371536`
- Best per-target SCCs observed:
  - WT: `0.8424858173749376`
  - ESP: `0.8199621805091523`
  - HF: `0.8123524358262405`
  - Sniper-Cas9: `0.4579662780166546`
  - HL60: `0.23627965491011052`
- Best WT->HL60 transfer SCC: `0.46050464169905975`

Frozen threshold gaps (`public_claim_thresholds.json`):
- mean9: `-0.16376181736284634`
- WT: `-0.018514182625062436`
- ESP: `-0.031037819490847718`
- HF: `-0.05264756417375949`
- Sniper-Cas9: `-0.4770337219833454`
- HL60: `-0.1657203450898895`
- WT->HL60: `-0.00749535830094028`

### Off-target (completed single-split and optuna frame runs)
- Best completed single-run LODO (held-out split) observed:
  - AUROC `0.9859604105571848`
  - AUPRC `0.9389281784201725`
- Best completed off-target Optuna study (nibi wave2):
  - Best trial AUROC `0.986721680420105`
  - Best trial AUPRC `0.9646667674048945`
  - Frame manifest: `data/public_benchmarks/off_target/frames/cclmoff_lodo.json`

Important comparability note:
- CIRCLE CV and CIRCLE->GUIDE claim frames remain blocked by unresolved blank `Method` bucket provenance in CCLMoff manifest policy.
- New `pub_off_sweep` jobs are now running to produce true all-split LODO aggregates rather than single held-out split snapshots.

## Files Added (this update)
- `scripts/run_public_off_target_manifest_sweep.py`
- `scripts/slurm_public_off_target_manifest_sweep.sh`
- `RUN_LOG_2026-03-05_CLUSTER_PARALLEL.md`

## Open Blockers
- beluga scheduler plugin mismatch blocks submission.
- cedar jobs queue but remain blocked by unavailable nodes.
- Uncertainty frame (`crispai_change_regression_uncertainty`) still blocked pending processed CHANGE-seq table staging.
- CIRCLE/CIRCLE->GUIDE claim frames remain blocked until blank `Method` provenance is resolved in primary CCLMoff mapping policy.
