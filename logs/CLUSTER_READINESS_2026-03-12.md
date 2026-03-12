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
