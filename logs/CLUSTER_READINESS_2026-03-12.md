# Cluster Readiness Snapshot — 2026-03-12

## Purpose
This note records the real execution readiness state of the accessible Alliance clusters for the CRISPR HNN/FMC benchmark workflow after SSH/env bootstrap and live Slurm probe tests.

## Ready for immediate follow-up work

### rorqual
- SSH persistence: yes
- repo: `/scratch/amird/chromaguide_experiments`
- env: `/scratch/amird/env_public_benchmark_hnn`
- scheduler: yes
- state: active benchmark target
- caution: v6 wave showed mixed completion plus repeated `CUDA_ERROR_NO_DEVICE`

### fir
- SSH persistence: yes
- repo: `/scratch/amird/chromaguide_experiments`
- env: `/scratch/amird/env_public_benchmark_hnn`
- scheduler path: `/opt/software/slurm/bin`
- explicit GPU-model held probe accepted: `27282317` then cancelled
- live TensorFlow GPU-check smoke submitted with explicit `--gpus=h100:1`: `27283173` (pending)
- note: generic partition selection works better than forcing `gpubase_interac`

### nibi
- SSH persistence: yes
- repo: `/scratch/amird/chromaguide_experiments`
- env: `/scratch/amird/env_public_benchmark_hnn`
- scheduler path: `/opt/software/slurm/24.11.7/bin`
- explicit GPU-model held probe accepted: `10195672` then cancelled
- live TensorFlow GPU-check smoke submitted with explicit `--gpus=h100:1`: `10195699` (pending)
- note: generic wrapper with `#SBATCH --gres=gpu:1` is not enough here; Nibi explicitly requires a GPU model for direct `sbatch` submissions

### narval
- SSH persistence: yes
- repo: `/scratch/amird/chromaguide_experiments`
- env: `/scratch/amird/env_public_benchmark_hnn`
- scheduler path: `/opt/software/slurm/bin`
- explicit GPU-model held probe accepted: `57703527` then cancelled
- wrapper smoke submitted: `57703551` (pending on unavailable nodes `ng[31205,31303]`)
- note: viable cluster, but the currently pending smoke needs either time or rerouting

## Partially prepared, but not currently GPU-ready for this workflow

### trillium
- SSH persistence: yes
- repo/env bootstrap: yes
- scheduler on CPU login works
- GPU job submission from current login path is not valid for this workflow
- Slurm errors showed:
  - valid allocation is `def-kwiese`, not `def-kwiese_gpu`
  - GPU submissions must come from the GPU login path (`trig-login01` / `trillium-gpu`)
- current `trillium-gpu` SSH path is not established yet
- conclusion: keep for CPU/support use until GPU login/account path is corrected

### killarney
- SSH persistence: yes
- repo/env bootstrap: yes
- scheduler binaries found at `/cm/shared/apps/slurm/current/bin`
- live Slurm config lookup still failed (`Could not establish a configuration source`)
- conclusion: not yet submission-ready

## Not usable under current account/path assumptions

### tamia
- SSH persistence: yes
- repo/env bootstrap: yes
- scheduler available
- GPU partitions visible
- direct submission failed: `Invalid account or account/partition combination specified`
- conclusion: blocked by account/allocation mismatch

### vulcan
- SSH persistence: yes
- repo/env bootstrap: yes
- scheduler available
- GPU partitions visible
- direct submission failed: `Invalid account or account/partition combination specified`
- conclusion: blocked by account/allocation mismatch

## Ignore for this workflow

### beluga
- reachable but low-value / decommission risk for heavy compute

### cedar
- reachable but low-value / operationally obsolete for this workflow

## Immediate execution guidance
1. Keep harvesting and diagnosing on `rorqual`.
2. Use `fir` and `nibi` as the next rerun targets if the current pending GPU-check smokes confirm real GPU visibility to TensorFlow.
3. Treat `narval` as usable but wait for the current smoke disposition before launching repaired reruns.
4. Do not schedule HNN GPU work on `trillium`, `killarney`, `tamia`, or `vulcan` until their current submission blockers are explicitly resolved.


## 2026-03-12 live submission probes
- `fir`: held GPU-model probe accepted (`27282317`, canceled immediately); live TensorFlow GPU-check smoke accepted as `27283173` and remains pending.
- `nibi`: held GPU-model probe accepted (`10195672`, canceled immediately); live TensorFlow GPU-check smoke accepted as `10195699` and remains pending.
- `narval`: held GPU-model probe accepted (`57703527`, canceled immediately); live HNN wrapper smoke accepted as `57703551` and is running on `ng10102`.
- `trillium`: current GPU path is blocked by login/account mismatch (`def-kwiese` vs `def-kwiese_gpu`, GPU jobs require GPU login path).
- `tamia` / `vulcan`: scheduler is reachable, but current Alliance account mapping is rejected for GPU submission.


## 2026-03-12 direct GPU diagnostics
- `fir`: submitted direct TensorFlow+nvidia-smi GPU diagnostic as job `27286411` after the earlier smoke showed `Could not find cuda drivers on your machine` despite H100 allocation.
- `narval`: submitted direct TensorFlow+nvidia-smi GPU diagnostic as job `57703637` after wrapper smoke `57703551` timed out with no summary.
- `nibi`: submitted direct TensorFlow+nvidia-smi GPU diagnostic as job `10195871`; the earlier GPU-check job `10195699` remains pending.


## 2026-03-12 direct GPU diagnostic outcomes
- `fir`:
  - direct diagnostic `27286411` failed due to a malformed heredoc in the `sbatch --wrap` payload, not a cluster/runtime problem.
  - corrected diagnostic `27286895` started on `fc10517` and proved the current TensorFlow environment still reports `Could not find cuda drivers on your machine` even with an allocated H100. This means `fir` is not yet a usable HNN rerun target without additional CUDA/runtime environment work.
- `narval`:
  - wrapper smoke `57703551` timed out after `00:10:26` with no `SUMMARY.json`.
  - first direct diagnostic `57703637` failed due to the same malformed heredoc payload, not a cluster/runtime problem.
  - corrected diagnostic `57703655` is pending with `ReqNodeNotAvail` and is the current gate for deciding whether `narval` is promotion-ready.
- `nibi`:
  - original direct diagnostic `10195871` was superseded and canceled.
  - corrected diagnostic `10195896` is pending with `ReqNodeNotAvail` and is the current gate for deciding whether `nibi` is promotion-ready.
- `rorqual`:
  - all original v6 jobs are done; no jobs remain in queue.
  - Group C summaries were harvestable for `s2024` and `s42`; `s220` remains on a flaky remote file path state (`Cannot send after transport endpoint shutdown`) and may need reconstruction from already observed metrics if the file remains unreadable.


## 2026-03-12 verbose GPU diagnostics in flight
- `fir`: verbose diagnostic `27290050` is running on `fc10417` with a 10-minute limit. It prints explicit stage markers (`START`, `AFTER_MODULE`, `nvidia-smi`, Python path, TensorFlow GPU list) and is the current gate for deciding whether `module load cuda/12.6` is sufficient to make the H100 visible.
- `narval`: verbose diagnostic `57703927` is running on `ng31002` with a 10-minute limit. It uses the same staged logging approach with `cuda/12.2` and is the current gate for deciding whether Narval is the first promotable rerun target.
- `nibi`: corrected diagnostic `10195896` remains pending with `ReqNodeNotAvail` and will be evaluated once it starts.


- follow-up from the verbose diagnostics: the staged logs proved `nvidia-smi` sees the GPU on both `fir` and `narval`, but the diagnostic still leaked the wrong Python path because the wrap payload allowed local shell expansion. Those jobs were superseded by explicit-venv-Python diagnostics to get a definitive TensorFlow GPU verdict.


## 2026-03-12 fixed-wrapper secondary smoke wave
- previous ad hoc direct diagnostics were superseded after hardening the wrappers and helper scripts.
- `fir`: fixed-wrapper HNN smoke submitted as `27292872`.
- `narval`: fixed-wrapper HNN smoke submitted as `57704059`.
- `nibi`: fixed-wrapper HNN smoke submitted as `10196105`.
- these are the first secondary-cluster smokes that exercise the real HNN wrapper path after removing the generic `#SBATCH --gres=gpu:1` conflict, using explicit cluster-appropriate `--gpus=...` requests and explicit `CUDA_MODULE` selection.
