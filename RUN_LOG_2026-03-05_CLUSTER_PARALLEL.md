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

## Checkpoint Update (2026-03-05 15:15 PST)

### Connection/Execution Health
- Persistent SSH masters verified live for: `nibi`, `rorqual`, `fir`, `cedar`, `beluga`, `trillium`.
- Scheduler probes:
  - `nibi`: active (`9838859` running)
  - `rorqual`: active (`7728224` running)
  - `fir`: active (`25783647`, `25786572`, `25788418` running)
  - `cedar`: queued only (node availability)
  - `trillium`: active CPU searches (`1119483`, `1119950`, `1124392`, `1124393`)
  - `beluga`: still blocked by Slurm plugin mismatch

### Newly Harvested Artifacts
Harvested into `results/public_benchmarks/cluster_harvest_20260305/`:
- `nibi/full_run_best_FINAL_SUMMARY.json`
- `nibi/off_target_manifest_sweep_summary.json`
- `nibi/optuna_off_target_wave2_summary.json`
- `rorqual/parallel_full_wave2_FINAL_SUMMARY.json`
- `rorqual/off_target_manifest_sweep_summary.json`
- `rorqual/optuna_off_target_wave2_summary.json`
- `fir/full_run_best_FINAL_SUMMARY.json`
- `fir/off_target_manifest_sweep_summary.json`
- `fir/optuna_off_target_wave2_summary.json`
- `trillium/optuna_on_target_summary.json`
- `trillium/optuna_off_target_summary.json`

### Current Metrics Delta
- On-target best completed mean9 SCC remains below claim threshold:
  - best: `0.5522381826371536` (nibi)
  - threshold: `0.716`
  - gap: `-0.16376181736284634`
- Best WT->HL60 transfer remains near but below target:
  - best: `0.46050464169905975`
  - target: `0.468`
  - gap: `-0.00749535830094028`
- Off-target all-split LODO now aggregated on 3 clusters:
  - nibi mean AUROC/AUPRC: `0.962681708886335 / 0.8828249823003957`
  - rorqual mean AUROC/AUPRC: `0.9652139708033906 / 0.8875998885739973`
  - fir mean AUROC/AUPRC: `0.9633986232867555 / 0.8842023967553448`

### Claim Status (unchanged)
- On-target: not claim-valid yet versus frozen thresholds.
- Off-target: LODO frame now robustly aggregated; CIRCLE/CIRCLE->GUIDE still blocked by unresolved blank `Method` provenance.
- Uncertainty/calibration: still blocked pending processed CHANGE-seq table staging.

## Checkpoint Update (2026-03-05 evening PST)

### Repo + Sync
- Pushed latest script updates to `origin/Dev` at commit `1c50143`.
- Synced updated `scripts/` to all cluster workdirs. Cedar required tar-over-SSH fallback because `rsync` is unavailable on login node.

### Compatibility Fixes Applied
- Fixed Python 3.9 incompatibility (`datetime.UTC`) by switching to `datetime.now(timezone.utc)` in:
  - `scripts/stage_change_seq_proxy_table.py`
  - `scripts/build_public_off_target_frames.py`

### Fresh Smoke Submissions (post-fix)
- nibi: `9844993` (off-target uncertainty smoke)
- rorqual: `7730448` (off-target uncertainty smoke)
- fir: `25800106` (GPU smoke pending), plus CPU fallback submission path validated
- cedar: `66523835` (off-target uncertainty smoke)
- trillium: `1126842` (off-target uncertainty smoke)
- beluga: submission blocked (plugin incompatibility)

### Full Parallel Wave Submitted
- nibi: `9845034` (on-target full), `9845036` (on-target optuna), `9845037` (off-target optuna LODO), `9845038` (off-target uncertainty full)
- rorqual: `7731085`, `7731105`, `7731133`, `7731166`
- fir: `25800913`, `25800915`, `25800916`, `25800917` (with `--partition=gpubase_bygpu_b2` routing)
- cedar: `66523848`, `66523849`, `66523850`, `66523851` (with `--partition=gpubase_bygpu_b2` routing)
- trillium: `1126846` (off-target uncertainty full CPU)
- beluga: full submission still blocked by plugin stack mismatch (`cc-tmpfs_mounts.so`)

### Real-Time Queue Snapshot Summary
- nibi: all new jobs pending on priority.
- rorqual: all new jobs pending on priority.
- fir: new full wave pending; existing `sota_fmc_pub` still running.
- cedar: new full wave pending with `ReqNodeNotAvail` constraint.
- trillium: existing CPU search jobs running; new smoke/full uncertainty jobs pending.
- beluga: no runnable queue path due scheduler plugin failure.

## Checkpoint Update (2026-03-05 late evening PST)

### Root Cause + Fixes Applied
- Fixed submission helper bug where `REPO_DIR=` was used for remote `cd` but not exported into the SLURM job env.
  - File: `scripts/submit_public_benchmark_cluster_jobs.sh`
  - Impact: wrappers previously fell back to `$HOME/chromaguide_experiments` (quota/old-path failures).
- Hardened uncertainty loader against malformed rows:
  - File: `scripts/train_public_off_target_uncertainty.py`
  - Change: skip rows with missing/bad `activity_log1p_read` and report skip counters.

### Corrective Resubmission Wave
After patching and syncing scripts to cluster workdirs, submitted corrected full wave with explicit scratch repo + venv:
- nibi: `9846669`, `9846670`, `9846671`, `9846672`
- rorqual: `7736323`, `7736345`, `7736376`, `7736394` then corrective resubmissions `7738771`, `7738791`, `7738824`
- fir: `25804396`, `25804397`, `25804399`, `25804400`
- cedar: `66524473`, `66524474`, `66524475`, `66524476`
- trillium: `1126921`
- beluga: still blocked by plugin mismatch

### Additional Data/Protocol Sync
- Generated canonical on-target folds locally (`scripts/build_public_on_target_folds.py`, k=5, seed=42).
- Synced folds + `public_claim_thresholds.json` to cluster scratch repos.
- Synced CCLMoff primary CSV into `rorqual:/scratch/.../primary_cclmoff/` to satisfy off-target/uncertainty frame prerequisites.

### Current Truth Snapshot
- The corrected job wave is now in queue/running across rorqual/fir/cedar/trillium, with nibi pending.
- rorqual already produced one clean completion in corrected wave context:
  - `7736376` (`pub_off_optuna`) -> `COMPLETED`.
- We still do **not** have claim-valid evidence of “beat all latest SOTA on every aspect” yet; we are still in execution/harvest phase.

### Addendum: nibi Runtime Stack Issue
- nibi corrected-wave job outcomes showed a Python stack mismatch in the scratch venv:
  - `ModuleNotFoundError: No module named 'torch.nn'` for on-target full and uncertainty jobs.
- Diagnostic on nibi scratch venv indicates `torch` resolves as a namespace package without `nn` submodule.
- Consequence:
  - nibi on-target/uncertainty jobs are currently blocked for valid training runs.
  - nibi optuna jobs that returned `-1.0` are exploratory-invalid and should not be used for promotion.
- Active workaround path: continue full-scale wave on `fir`, `rorqual`, `cedar`, and `trillium` while nibi Python stack is corrected.
