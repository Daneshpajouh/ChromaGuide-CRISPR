# Full Context Handover

Date: 2026-03-11  
Repo: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR`  
Branch: `Dev`  
HEAD at handover creation base: `f4a6132b3056b85a277c33cc4dc0297a815b2c29`

## 1. What This Document Is For

This is the full working handover for the public-benchmark and SOTA-comparison effort in this repository.

It is intended to let another engineer continue from zero context.

It covers:

- the project goal and claim rules
- the full trajectory from the early ChromaGuide branch state to the current public SOTA benchmarking phase
- what has been tried, what worked, what failed, and what should not be repeated
- exact local paths, cluster paths, environments, and commands
- the public artifact inspection findings
- the current benchmark status and live jobs
- the precise remaining blockers to a claim-valid “we outperform all SOTA across all aspects” statement

This document should be read together with the machine-readable ledgers:

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/PUBLIC_EXECUTION_STATUS_2026-03-05.json`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_UPSTREAM_REPRO_STATUS_2026-03-05.json`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/CLUSTER_BENCHMARK_EXECUTION_STATUS_2026-03-05.json`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-10.json`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-10.md`

## 2. Executive Summary

Current exact position:

- `On-target`: still below frozen claim-valid SOTA thresholds on every strict row.
- `Off-target primary`: still blocked by CCLMoff provenance, not by model quality.
- `Uncertainty`: exact public-bundle topology now numerically clears the frozen target, but final claim-valid attribution is still not frozen.
- `DeepHF`: canonical public artifacts are now pinned and usable.
- `CRISPR_HNN` and `CRISPR-FMC`: public retrain repos exist, but public frozen checkpoint parity is still incomplete.

The nearest remaining strict on-target miss is:

- `WT->HL60 SCC = 0.4653505491794053` vs target `0.468`, gap `-0.002649450820594701`

The current exact strict scoreboard is:

| Metric | Best | Target | Gap (best-target) | Pass |
|---|---:|---:|---:|---|
| on_target mean9 SCC | 0.6743715317106037 | 0.716 | -0.04162846828939626 | No |
| WT SCC | 0.8453855978444285 | 0.861 | -0.015614402155571527 | No |
| ESP SCC | 0.8361978319569829 | 0.851 | -0.014802168043017039 | No |
| HF SCC | 0.8383885437299339 | 0.865 | -0.026611456270066114 | No |
| Sniper-Cas9 SCC | 0.9274682254601416 | 0.935 | -0.007531774539858427 | No |
| HL60 SCC | 0.3130585941476802 | 0.402 | -0.0889414058523198 | No |
| WT→HL60 SCC | 0.4653505491794053 | 0.468 | -0.002649450820594701 | No |
| uncertainty CHANGE-seq test Spearman | 0.5119756727197794 | 0.5114 | +0.0005756727197794298 | Yes |

Important caveat:

- the uncertainty row above is still `claim_valid = false`
- off-target primary remains unavailable / blocked
- therefore we still do **not** have a claim-valid “beat all SOTA in all aspects” result

## 3. Claim Rules

These rules have been applied strictly throughout the public benchmarking phase.

### 3.1 Claim-valid means

A metric is only claim-valid if:

- the frame is source-backed and frozen
- the split regime matches the claimed benchmark
- the metric is computed on the correct target column / topology
- the artifact provenance is clear enough to defend publicly

### 3.2 Not claim-valid means

A metric is not claim-valid if any of the following hold:

- the evaluation frame is a proxy or reconstruction rather than the frozen published frame
- source dataset provenance is ambiguous
- the public artifact surface does not expose the needed split or method mapping
- the exact metric definition is still being reverse-engineered

### 3.3 Current implications

- `On-target` strict rows are claim-validly scored.
- `Off-target primary` strict rows are blocked and must remain unavailable.
- `Uncertainty` is numerically above target under the corrected exact-bundle topology, but still not claim-valid until the paper metric definition is frozen exactly.

## 4. Repository and Working Layout

### 4.1 Local repository root

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR`

### 4.2 Main status / report / benchmark files

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/PUBLIC_EXECUTION_STATUS_2026-03-05.json`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_UPSTREAM_REPRO_STATUS_2026-03-05.json`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/CLUSTER_BENCHMARK_EXECUTION_STATUS_2026-03-05.json`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-10.json`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-10.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/public_claim_thresholds.json`

### 4.3 Public benchmark source roots

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/data/public_benchmarks/sources/CCLMoff`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/data/public_benchmarks/sources/CRISPR_HNN`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/data/public_benchmarks/sources/CRISPR-FMC`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/data/public_benchmarks/sources/DeepHF`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/data/public_benchmarks/sources/crispAI_crispr-offtarget-uncertainty`

### 4.4 Exact uncertainty bundle paths

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/data/public_benchmarks/off_target/crispai_parity/data_bundle/`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/data/public_benchmarks/off_target/crispai_parity/model_bundle/`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/data/public_benchmarks/off_target/crispai_parity/crispai_parity_manifest.json`

### 4.5 Recent harvested result roots

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/results/public_benchmarks/cluster_harvest_20260308/rorqual/`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/results/public_benchmarks/cluster_harvest_20260309/rorqual/`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/results/public_benchmarks/cluster_harvest_20260311/rorqual/`

## 5. High-Level History: From Early Branch State To Current SOTA Work

The repository history splits into three main eras.

### 5.1 Era A: Original ChromaGuide development and internal proposal optimization

This phase predates the public SOTA benchmarking layer.

Relevant earlier commits include:

- `737e3a2` Clean repo with DNABERT-Mamba pipeline and Narval results
- `741fe65` Unified ChromaGuide V2 training pipeline
- `14112a9` Narval SLURM script for DeepHF training
- `e97170b` Narval SLURM script for CRISPRon training
- `e4820ec` Evaluation framework - statistical testing and SOTA comparison
- `2adb664` Backbone ablation framework
- `b668acc` Architecture modules including conformal prediction, off-target, design score aggregator

The branch was initially focused on:

- internal on-target optimization
- off-target model capacity work
- multimodal / DNABERT-2 / Mamba integration
- internal proposal thresholds and statistical validation

At that stage the branch had strong internal progress but was still mixing:

- internal optimization success
- partially comparable external references
- non-frozen evaluation regimes

### 5.2 Era B: Late-February internal push to exceed proposal-level thresholds

This phase is captured in:

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/RUN_LOG_2026-02-25_2026-02-26.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_AUDIT_AND_BENCHMARK_MASTER_PLAN_2026-03-02.md`

Key outcomes from that phase:

- stacked / OOF on-target metrics became extremely strong
- off-target matched held-out performance saturated high AUROC / AUPRC
- conformal coverage and significance checks passed
- non-stacked claim-valid internal on-target threshold was still a near miss

Important exact findings from that phase:

- best stacked/OOF on-target values exceeded internal proposal thresholds comfortably
- best non-stacked claim-valid ensemble was still slightly under target
- strict Split-A regime remained far below the 0.911-style threshold

This phase matters because it established the discipline later reused in the public SOTA phase:

- separate claim-valid from merely strong internal results
- keep stacked / OOF / strict held-out / transfer results separate
- do not round near misses into passes

### 5.3 Era C: March public-benchmark and external-SOTA hardening

This is the current phase and the most relevant one for continuation.

This phase introduced:

- frozen public claim thresholds in `public_claim_thresholds.json`
- machine-readable public status ledgers
- cluster-wide public benchmark orchestration
- upstream public source acquisition and smoke repro
- strict scoreboard generation
- all-aspects reporting
- artifact inspection and claim-validity audits

Representative milestone commits:

- `79d623f` Add SOTA source registry and acquisition reproducibility pipeline
- `5ce12cf` Add SOTA source smoke reproducibility artifact
- `069e1a8` Add upstream SOTA baseline runtime repro runners and status artifacts
- `b368c23` Add strict SOTA scoreboard generator
- `a6dc450` Add 2026-03-06 strict SOTA scoreboard
- `18da22f` Add all-aspects scoreboard markdown and upstream HNN wrapper
- `1e4ca91` Add HNN transfer benchmark path and log FMC canonical9
- `8a1605a` Harvest HNN canonical9 results and launch tuned rerun
- `e4bce4a` Harvest tuned HNN runs and refresh strict scoreboard
- `018dec6` Add deep research prompt pack
- `25d1b84` Add final artifact inspection prompt
- `b15de37` Add artifact inspection script and initial manifests
- `bd39625` Improve artifact inspection with Docker registry fallback
- `c90f647` Log crispAI parity bundle inspection
- `22f8884` Add crispAI parity frame extractor
- `1adfd28` Log crispAI exact-bundle parity attempt
- `a26c762` Update uncertainty topology diagnosis and DeepHF audit
- `da88c69` Log March 10 targeted rorqual SOTA wave
- `f4a6132` Log FMC v2 result and corrected HNN reruns

## 6. Frozen Public Targets

The public target file is:

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/public_claim_thresholds.json`

Important frozen values:

### 6.1 On-target

Global average target:

- `mean_SCC_9_dataset = 0.716`
- `mean_PCC_9_dataset = 0.721`

Per-dataset SCC targets:

- WT `0.861`
- ESP `0.851`
- HF `0.865`
- Sniper-Cas9 `0.935`
- HL60 `0.402`

Transfer target:

- WT→HL60 SCC `0.468`
- WT→HL60 PCC `0.459`

### 6.2 Off-target primary

- CIRCLE-seq CV AUROC `0.985`
- CIRCLE-seq CV AUPRC `0.524`
- CIRCLE→GUIDE AUROC `0.996`
- CIRCLE→GUIDE AUPRC `0.520`
- DIG-seq LODO AUROC `0.985`
- DIG-seq LODO AUPRC `0.720`
- DISCOVER-seq+ LODO AUROC `0.944`
- DISCOVER-seq+ LODO AUPRC `0.665`

### 6.3 Uncertainty

- CHANGE-seq test Spearman `0.5114`

## 7. Exact Current Status

### 7.1 Strict scoreboard now

Source files:

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-10.json`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-10.md`

Current exact rows:

| Metric | Best | Target | Gap (best-target) | Pass |
|---|---:|---:|---:|---|
| on_target mean9 SCC | 0.6743715317106037 | 0.716 | -0.04162846828939626 | No |
| WT SCC | 0.8453855978444285 | 0.861 | -0.015614402155571527 | No |
| ESP SCC | 0.8361978319569829 | 0.851 | -0.014802168043017039 | No |
| HF SCC | 0.8383885437299339 | 0.865 | -0.026611456270066114 | No |
| Sniper-Cas9 SCC | 0.9274682254601416 | 0.935 | -0.007531774539858427 | No |
| HL60 SCC | 0.3130585941476802 | 0.402 | -0.0889414058523198 | No |
| WT→HL60 SCC | 0.4653505491794053 | 0.468 | -0.002649450820594701 | No |
| uncertainty CHANGE-seq test Spearman | 0.5119756727197794 | 0.5114 | +0.0005756727197794298 | Yes |

### 7.2 Off-target primary status now

All primary rows remain unavailable.

Reason:

- unresolved method provenance in the public CCLMoff frame
- no claim-valid relabeling allowed
- therefore no claim-valid CIRCLE / CIRCLE→GUIDE / DIG / DISCOVER+ primary scores can be published from the current public artifact set

### 7.3 Current live jobs now

Latest checked on `rorqual`:

- `8032850` `sota_hnn_pub` `RUNNING`
- `8032851` `sota_hnn_xfer` `COMPLETED`

Exact transfer v5 completion:

- run tag: `sota_hnn_transfer_wt_hl60_v5_bs16_lr2e4_p8_e40_envfix`
- summary file:
  - `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/results/public_benchmarks/cluster_harvest_20260311/rorqual/sota_hnn_transfer_wt_hl60_v5_bs16_lr2e4_p8_e40_envfix_SUMMARY.json`
- mean SCC: `0.4585020062225119`
- gap vs frozen `0.468`: `-0.009497993777488105`
- outcome: did not improve the standing WT→HL60 best

Observed partial canonical9 v5 signal before handover:

- `WT_fold0.json = 0.8498275218185867`
- `WT_fold1.json = 0.8490297301978146`
- partial WT mean over 2 folds = `0.8494286260082007`

Interpretation:

- better than the standing WT best so far
- still below WT target
- not strict-board relevant until the full run completes and all datasets are harvested

## 8. What Has Been Tried

This section matters because the same dead ends should not be repeated blindly.

### 8.1 Internal on-target architecture exploration before public SOTA phase

Earlier branch work explored:

- DNABERT-2 multimodal paths
- CNN-GRU backbones
- Mamba-like sequence models
- cross-attention multimodal fusion
- large synthetic / Narval-backed training waves
- stacked and OOF ensembling

Outcome:

- excellent internal optimization results
- not directly sufficient for frozen public SOTA claims
- some old artifacts were strong but not directly comparable to current public benchmark regimes

### 8.2 March public on-target benchmark waves

Major families executed:

- public on-target full runs
- public on-target Optuna waves
- CPU fallbacks on cedar / trillium
- gate / cross-attention / mamba variants in earlier cluster-first waves
- upstream public replications of HNN and FMC
- targeted HNN reruns focused on:
  - canonical9 mean9 improvement
  - WT→HL60 transfer
  - near-gap flipping on WT / ESP / Sniper

Representative observations:

- early public mean9 scores were around `0.552`
- then improved through several waves into the `0.56x`, then `0.65x`, then `0.6743715317106037`
- HNN consistently outperformed FMC in our matched completed public reruns
- HNN improved Sniper materially into the `0.92x` regime
- WT→HL60 repeatedly came close but has not flipped the frozen threshold

### 8.3 Public FMC matched runs

Several FMC runs were attempted.

Important harvested results include:

- threshold-set / partial runs that underperformed HNN
- canonical9 v1 and v2 reruns
- large-batch protocol-aligned FMC v2 on `rorqual`

Most recent exact FMC v2 result:

- file:
  - `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/results/public_benchmarks/cluster_harvest_20260311/rorqual/sota_fmc_canonical9_v2_bs4096_e30_s2024_SUMMARY.json`
- mean9 `0.5945087755724462`
- WT `0.7404842039870346`
- ESP `0.7371299436354073`
- HF `0.7174789739321505`
- Sniper `0.9043398453158235`
- HL60 `0.2545666123071556`

Conclusion:

- this FMC v2 protocol-aligned rerun did not improve the strict board
- public FMC retrain repo is useful for experiments, but in our runs it has not yet beaten the best HNN-based results

### 8.4 Public HNN runs

HNN has produced the strongest public on-target results so far in this repo.

Important milestones:

- threshold-set HNN runs improved ESP, HF, Sniper, HL60 relative to earlier baselines
- canonical9 tuned HNN waves improved mean9 into the `0.6743` range
- transfer HNN waves repeatedly came close on WT→HL60

Recent job sequence:

- `8005698` HNN transfer v4 failed due wrong env
- env fix applied: dedicated `env_public_benchmark_hnn`
- reruns submitted:
  - `8032850` canonical9 v5
  - `8032851` transfer v5
- `8032851` completed below target
- `8032850` still running at last check

### 8.5 Off-target public waves

Off-target execution included:

- single-split runs in early waves
- corrected all-split manifest sweep runners
- off-target LODO aggregate runs across multiple clusters
- CCLMoff runtime smokes and import-forward validation
- proxy/fallback frames when primary provenance was blocked

Key historical pivot:

- early manifest-based off-target jobs were only hitting a single held-out split per job
- this was corrected by adding the manifest sweep runner so all splits were aggregated

Representative files:

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/scripts/run_public_off_target_manifest_sweep.py`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/scripts/slurm_public_off_target_manifest_sweep.sh`

Outcome:

- robust proxy / fallback off-target metrics were produced
- primary CCLMoff claim frames remain blocked

### 8.6 Uncertainty / crispAI work

This has gone through several distinct stages.

#### Stage 1: runtime unblock

Initial blockers included:

- dependency issues
- genome assets
- compatibility shims
- annotation preprocessing edge cases

These were gradually fixed.

#### Stage 2: exact-bundle acquisition

We then downloaded and inspected the exact public bundles.

Important artifacts:

- data bundle
- model bundle
- checkpoint `epoch:19-best_valid_loss:0.270.pt`
- exact public train/test CSVs

#### Stage 3: stale parity attempt

An earlier exact-bundle evaluation path produced:

- mean Spearman `0.45398406235826966`
- median `0.33062395802954836`

That path is now understood to be stale / wrong-topology for the frozen target.

#### Stage 4: topology diagnosis

A corrected topology diagnosis showed:

- exact-bundle raw-mean on `CHANGEseq_reads_adjusted`
- Spearman `0.5119756727197794`
- target `0.5114`
- numerical pass by `+0.0005756727197794298`

Artifact:

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/results/public_benchmarks/crispai_parity_topology_diagnosis.json`

Current conclusion:

- the crispAI issue is no longer asset discovery
- it is metric-definition freezing and claim attribution discipline

## 9. Current Methodology By Area

### 9.1 On-target methodology

Current public on-target comparisons are based on frozen thresholds and claim-valid rows tracked in the scoreboard.

Key regimes:

- canonical9 mean SCC
- per-dataset SCC
- WT→HL60 transfer SCC

Practical operating rules:

- treat completed full-run summaries as the source of truth
- do not promote partial folds into strict scoreboard rows
- do not mix internal older ChromaGuide metrics with public SOTA rows
- treat DeepHF canonical artifacts as the source-backed anchor for WT / ESP / HF comparability

### 9.2 Off-target methodology

Two very different layers exist:

#### Primary claim frames

These are the frozen CCLMoff-style frames:

- CIRCLE CV
- CIRCLE→GUIDE
- DIG LODO
- DISCOVER+ LODO

Current status:

- blocked by missing method provenance
- must remain unavailable

#### Fallback / proxy frames

These are still useful operationally for model development, but not for primary claim language.

Rules:

- if used, label them explicitly fallback or proxy
- do not let them silently replace the primary frames in reporting

### 9.3 Uncertainty methodology

Rules learned the hard way:

- do not trust a numerically good result unless the evaluator topology is nailed down
- preserve the exact public-bundle train/test and checkpoint paths
- keep direct array-level and column-level diagnostics around
- treat stale Box-Cox/jitter paths as historical debug branches, not the main parity evaluator

### 9.4 DeepHF methodology

Use DeepHF for:

- canonical artifact anchoring of WT / ESP / HF
- checking whether our public retrain paths are aligned to the actual public source assets

Do not assume:

- sequence-only CSVs are enough to match DeepHF
- because DeepHF canonical artifacts include biological features and specific environment assumptions

## 10. Public Artifact Inspection Findings

This section consolidates the artifact and browser work.

### 10.1 CCLMoff

Public surfaces inspected:

- repo
- Figshare
- Docker Hub
- paper / supplement surfaces

Exact browser-visible finding:

- Figshare only exposes:
  - `09212024_CCLMoff_dataset.csv`
  - `CCLMoff_V1.ckpt`

Local direct Docker inspection found:

- `/workspace` only mirrors the public code tree
- no hidden method map or split manifest in the inspected image workspace

Final operational verdict:

- CCLMoff primary off-target remains blocked
- no browser-visible or directly inspected public artifact has yet unblocked the method provenance problem

### 10.2 crispAI

Public artifact findings:

- repo is public
- bundle artifacts are public and staged locally
- relevant code exists for inference and model definition
- no public browser-visible standalone script definitively documenting the `0.5114` evaluation topology was found

Operational conclusion:

- the correct path is to preserve our exact topology diagnosis and explicitly distinguish it from the stale evaluator

### 10.3 CRISPR_HNN

Browser / artifact conclusion:

- retrain-capable public repo
- dataset CSVs present
- frozen checkpoint deleted from public repo
- no release assets / tags / pretrained frozen artifact path

Operational implication:

- good for retraining and architecture recovery
- insufficient for frozen-checkpoint parity claims

### 10.4 CRISPR-FMC

Browser / artifact conclusion:

- retrain-capable public repo
- dataset CSVs present
- architecture code present
- no public frozen pretrained model or release asset

Operational implication:

- useful for retraining and protocol-matching
- still retrain-only, not frozen-checkpoint parity capable

### 10.5 DeepHF

Browser / artifact conclusion:

- public `.pkl` data present
- public `.hd5` models present
- notebooks and utilities present
- this is the cleanest canonical public source among the on-target upstream sources

Operational implication:

- use it as the reference anchor for WT / ESP / HF alignment

### 10.6 Comet browser report

An external browser-based inspection report was also produced and should be preserved as supporting evidence.

Google Doc:

- [Comet Artifact Inspection Report](https://docs.google.com/document/d/1JDv5eLKT5FoNG8Zp_nSqytnSIjxxcsu3XD9fMrbKQ1o/edit)

Most important outcomes from that report:

- `CCLMoff`: exactly two public Figshare files were visible, reinforcing the primary-frame provenance block
- `crispAI`: browser surfaces did not expose a dedicated script clearly reproducing the published `0.5114` path
- `CRISPR_HNN`: public dataset CSV headers were confirmed browser-visible; deleted `.h5` checkpoint was noted
- `CRISPR-FMC`: public dataset CSV headers were confirmed browser-visible; architecture/train scripts confirmed
- `DeepHF`: public `.pkl` and `.hd5` artifacts were confirmed browser-visible and complete

## 11. Cluster Operating Guide

### 11.1 Clusters used in this project

The public benchmark phase has touched:

- `nibi`
- `rorqual`
- `fir`
- `cedar`
- `trillium`
- `beluga`

Current practical preference:

- `rorqual` is the most reliable target for the upstream public HNN/FMC work in the current phase

### 11.2 Basic SSH

Typical commands:

```bash
ssh rorqual
ssh nibi
ssh fir
ssh cedar
ssh trillium
```

Notes:

- `nibi` can trigger Alliance MFA and has been less convenient in recent public-benchmark work
- `fir` has had instability and earlier scheduler / time-limit issues
- `beluga` was blocked by plugin issues in earlier waves

### 11.3 Remote working repo path

The active remote working path used recently on `rorqual` is:

```bash
/scratch/amird/chromaguide_experiments
```

Important:

- this is effectively a synced working directory
- it is not treated as the authoritative git checkout
- sync is usually done by tar-over-ssh for specific script subsets

### 11.4 Remote environments

Generic benchmark env:

```bash
/scratch/amird/env_public_benchmark
```

Dedicated HNN env:

```bash
/scratch/amird/env_public_benchmark_hnn
```

Critical lesson:

- HNN jobs must explicitly pin the HNN env
- otherwise `keras_multi_head` may be missing and jobs fail

### 11.5 Querying queue state

```bash
ssh rorqual 'squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.20R"'
```

Useful narrower variant:

```bash
ssh rorqual 'squeue -u $USER | egrep "8032850|8032851" || true'
```

### 11.6 Syncing scripts to remote scratch repo

Typical pattern used recently:

```bash
tar -cf - \
  scripts/slurm_sota_crispr_fmc_public.sh \
  scripts/run_sota_crispr_fmc_public.py \
  scripts/slurm_sota_crispr_hnn_public.sh \
  scripts/run_sota_crispr_hnn_public.py \
  scripts/slurm_sota_crispr_hnn_transfer_public.sh \
  scripts/run_sota_crispr_hnn_transfer_public.py \
  scripts/submit_public_benchmark_cluster_jobs.sh \
| ssh rorqual 'cd /scratch/$USER/chromaguide_experiments && tar -xf -'
```

This was used because:

- rsync availability varied by system
- small targeted syncs were faster and more reliable than whole-repo syncs

### 11.7 Submitting jobs

Helper script used repeatedly:

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/scripts/submit_public_benchmark_cluster_jobs.sh`

Representative submission patterns:

```bash
ssh rorqual 'cd /scratch/amird/chromaguide_experiments && bash scripts/submit_public_benchmark_cluster_jobs.sh'
```

and direct `sbatch` when needed.

### 11.8 Harvesting results

Representative pattern:

```bash
scp rorqual:/scratch/amird/chromaguide_experiments/results/public_benchmarks/<run_tag>/SUMMARY.json \
  /Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/results/public_benchmarks/cluster_harvest_YYYYMMDD/rorqual/<run_tag>_SUMMARY.json
```

### 11.9 What to do after each harvest

Always do these steps in order:

1. copy the summary locally into the dated harvest directory
2. update `PUBLIC_EXECUTION_STATUS_2026-03-05.json`
3. update `CLUSTER_BENCHMARK_EXECUTION_STATUS_2026-03-05.json`
4. update `SOTA_UPSTREAM_REPRO_STATUS_2026-03-05.json` if upstream repro state changed
5. regenerate scoreboard if a strict row might have changed
6. commit and push

## 12. Git And Results Hygiene

### 12.1 Branch

- work is currently on `Dev`

### 12.2 Important result files are often ignored

When adding harvested result summaries under `results/`, it is sometimes necessary to force-add them.

Typical pattern:

```bash
git add -f results/public_benchmarks/cluster_harvest_20260311/rorqual/*.json
```

### 12.3 Safe workflow used repeatedly

```bash
git status --short
git add <files>
git commit -m "<clear message>"
git push origin Dev
```

### 12.4 Do not do this

- do not hard reset the repo
- do not rewrite history
- do not treat partial cluster outputs as final board moves
- do not silently overwrite status ledgers without corresponding harvested artifact evidence

## 13. Key Historical Documents To Read If Deeper Context Is Needed

These documents capture earlier parts of the effort and should be treated as supporting references.

### 13.1 Core historical run logs and plans

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/RUN_LOG_2026-02-25_2026-02-26.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/RUN_LOG_2026-03-05_CLUSTER_PARALLEL.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_AUDIT_AND_BENCHMARK_MASTER_PLAN_2026-03-02.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SOURCE_REPRO_LOG_2026-03-05.md`

### 13.2 Prompt and research artifacts

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/DEEP_RESEARCH_PROMPTS_2026-03-09.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/FOLLOWUP_DEEP_RESEARCH_PROMPT_2026-03-09.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/FOLLOWUP_DEEP_RESEARCH_PROMPT_2_2026-03-09.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/FINAL_ARTIFACT_INSPECTION_PROMPT_2026-03-09.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/TARGETED_DEEP_RESEARCH_PROMPT_2026-03-10.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/IMPLEMENTATION_DEEP_RESEARCH_PROMPT_2026-03-10.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/COMET_BROWSER_ARTIFACT_INSPECTION_PROMPT_2026-03-10.md`

### 13.3 Legacy repo-level summaries and deployment docs

- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/CRITICAL_FIX_SUMMARY.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/DEPLOYMENT_REPORT_FINAL.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/FIR_DEPLOYMENT_GUIDE.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/NARVAL_DEPLOYMENT.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/PUSH_AND_LOG_PROTOCOL.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/TRAINING_RESULTS_SUMMARY.md`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/LATEST_STATUS_2026-02-26.md`

## 14. Exact Findings By Area

### 14.1 On-target findings

- HNN has been the strongest public retrain family in our hands.
- FMC protocol-aligned retraining has not yet matched the best HNN public runs.
- WT→HL60 remains the closest frozen gap.
- HL60 remains the dominant drag on mean9.
- DeepHF canonical assets should be used as the reference anchor for WT / ESP / HF.

### 14.2 Off-target findings

- LODO and proxy work can be done operationally.
- claim-valid primary off-target comparison is blocked by CCLMoff provenance.
- no public artifact inspection so far has unblocked that.

### 14.3 Uncertainty findings

- exact public bundle is present and sufficient for strong diagnosis.
- the stale under-target result came from the wrong evaluator path.
- a numerically target-clearing exact-bundle path exists now.
- claim validity still depends on freezing exact paper attribution cleanly.

### 14.4 Upstream-public-source findings

- `CRISPR_HNN`: retrain repo usable, frozen checkpoint parity incomplete
- `CRISPR-FMC`: retrain repo usable, frozen checkpoint parity incomplete
- `DeepHF`: canonical artifacts present and pinned
- `CCLMoff`: ckpt public, provenance insufficient for primary-frame reconstruction
- `crispAI`: bundle public, parity now an evaluator-definition problem

## 15. Known Failure Modes And Lessons

### 15.1 HNN environment mismatch

Failure observed:

- transfer wrapper accidentally used generic env
- missing `keras_multi_head`
- job failed

Fix:

- explicitly set `VENV_DIR=/scratch/amird/env_public_benchmark_hnn`

### 15.2 Cluster submission helper export issues

Failure observed earlier:

- comma-valued env vars were truncated in Slurm export paths
- multi-dataset upstream runs could collapse incorrectly

Fix:

- submission helper hardened so comma-valued env vars are preserved correctly

### 15.3 CCLMoff comparability trap

Failure mode:

- seeing strong proxy off-target metrics and accidentally treating them as primary-frame parity

Rule:

- never do this
- keep primary off-target unavailable unless provenance is solved

### 15.4 crispAI evaluator trap

Failure mode:

- treating a stale Box-Cox/jitter or wrong-topology evaluator as the parity answer

Rule:

- preserve topology diagnosis artifacts
- do not regress to the stale `0.453984...` path as if it were final

### 15.5 Partial fold temptation

Failure mode:

- wanting to promote partial fold signals to the strict board

Rule:

- do not do this
- partial fold signals are operational guidance only

## 16. Exact Next Steps

These are the correct next actions in priority order.

### 16.1 Immediate

1. harvest `8032850` as soon as it completes
2. update the three machine-readable ledgers
3. regenerate the strict scoreboard if any row moves

### 16.2 If `8032850` improves strict rows materially

- commit harvested artifacts
- refresh `SOTA_SCOREBOARD_2026-03-10.*` or generate a new date-stamped scoreboard snapshot
- report exact deltas versus frozen thresholds

### 16.3 If `8032850` does not flip near gaps

Then the next realistic actions are:

1. run another HNN focused search only if the gap profile still justifies it
2. otherwise prioritize:
   - exact uncertainty claim freezing
   - DeepHF-grounded on-target audit
   - stronger HL60-focused fine-tuning or transfer design

### 16.4 Off-target

Do not spend more modeling cycles trying to publish primary off-target parity until a real public method map exists.

Allowed:

- fallback / proxy off-target runs for internal selection

Not allowed:

- claim-valid primary CCLMoff parity language

## 17. What Someone New Should Actually Do First

If taking over cold, do this exact sequence.

1. read:
   - this file
   - `PUBLIC_EXECUTION_STATUS_2026-03-05.json`
   - `SOTA_SCOREBOARD_2026-03-10.json`
   - `SOTA_UPSTREAM_REPRO_STATUS_2026-03-05.json`
2. check live jobs:

```bash
ssh rorqual 'squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.20R" | egrep "8032850|8032851" || true'
```

3. if `8032850` is complete, harvest its summary
4. update ledgers
5. only then decide whether the next action is another HNN run, uncertainty finalization, or a scoreboard freeze

## 18. Bottom Line

As of this handover:

- we have strong infrastructure, strong provenance tracking, and real public benchmark evidence
- we do **not** yet have a claim-valid “beat all SOTA in all aspects” result
- the closest on-target gap is still WT→HL60
- the main hard science gap is HL60 and mean9
- the main hard claim blocker is CCLMoff primary provenance
- the main uncertainty task is no longer artifact discovery but exact metric-freezing

That is the honest current state.
