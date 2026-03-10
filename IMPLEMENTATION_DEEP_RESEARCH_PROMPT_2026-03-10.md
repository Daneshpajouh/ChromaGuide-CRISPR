# Implementation Deep Research Prompt: Return Full Code, Files, and Execution Plan

Use this prompt exactly as written.

## Hard-Fail Instruction
Do not return a partial result.
Do not return only analysis.
Do not return only papers.
Do not return only recommendations.
Do not return pseudocode.

I need one consolidated output that contains:
- exact root-cause summary
- exact implementation plan
- exact file tree to add or modify
- exact code for every new file needed
- exact patches for every existing file that must change
- exact commands to download, stage, preprocess, train, evaluate, and summarize
- exact benchmark manifests and split definitions
- exact cluster launch commands and job specs
- exact experiment matrix with promotion/kill rules
- exact claim-safe reporting logic

If you cannot provide full code for a file, explicitly say why and provide the smallest complete runnable substitute.
If any required code file, manifest, command, or evaluation step is missing, the task is incomplete.

## Objective
We are trying to outperform the latest CRISPR SOTA in a claim-valid way across:
- on-target
- off-target primary
- uncertainty/calibration
- integrated design

The goal is not another survey. The goal is to produce the exact implementation package we can execute immediately.

## Current Exact State
### Strict on-target scoreboard
| Metric | Best | Target | Gap (best-target) | Pass |
|---|---:|---:|---:|---|
| on_target mean9 SCC | 0.6743715317 | 0.716 | -0.0416284683 | No |
| WT SCC | 0.8453855978 | 0.861 | -0.0156144022 | No |
| ESP SCC | 0.8361978320 | 0.851 | -0.0148021680 | No |
| HF SCC | 0.8383885437 | 0.865 | -0.0266114563 | No |
| Sniper-Cas9 SCC | 0.9274682255 | 0.935 | -0.0075317745 | No |
| HL60 SCC | 0.3130585941 | 0.402 | -0.0889414059 | No |
| WT->HL60 SCC | 0.4653505492 | 0.468 | -0.0026494508 | No |

### Uncertainty exact-bundle result
| Metric | Best | Target | Gap (best-target) | Pass |
|---|---:|---:|---:|---|
| crispAI exact-bundle test Spearman | 0.4539840624 | 0.5114 | -0.0574159376 | No |
| crispAI exact-bundle test Spearman (median) | 0.3306239580 | 0.5114 | -0.1807760420 | No |

### Current hard blockers
1. **CCLMoff primary off-target remains claim-invalid**
- Public Figshare currently exposes only:
  - `09212024_CCLMoff_dataset.csv`
  - `CCLMoff_V1.ckpt`
- Blank `Method` bucket has `3,282,935` rows.
- This is consistent with collapsed CIRCLE-seq + bulged GUIDE-seq.
- Docker inspection found no hidden mapping or split manifest in `/workspace`.
- Therefore claim-valid CIRCLE / GUIDE / LODO primary frames remain blocked.

2. **crispAI is now a reproduction-divergence problem**
- Exact Zenodo train CSV staged.
- Exact Zenodo test CSV staged.
- Exact best checkpoint staged: `epoch:19-best_valid_loss:0.270.pt`
- Exact-bundle rerun fixed row count to `168465`.
- But exact-bundle reproduced Spearman is `0.4539840624`, not `0.5114`.

3. **On-target is close on several rows but still below strict thresholds**
- Closest gaps:
  - `WT->HL60`: `-0.0026494508`
  - `Sniper-Cas9`: `-0.0075317745`
  - `ESP`: `-0.0148021680`
  - `WT`: `-0.0156144022`
- Largest remaining gaps:
  - `HL60`: `-0.0889414059`
  - `mean9`: `-0.0416284683`

4. **Public repo parity for HNN/FMC is partial**
- CRISPR_HNN public repo: train code + datasets present, but no public checkpoints/releases pinned.
- CRISPR-FMC public repo: train code + datasets present, but no public checkpoints/releases pinned.
- DeepHF canonical artifacts are pinned and available.

### Canonical DeepHF artifacts pinned
- `data/wt_seq_data_array.pkl`
- `data/esp_seq_data_array.pkl`
- `data/hf_seq_data_array.pkl`
- `models/DeepWt.hd5`
- `models/DeepWt_T7.hd5`
- `models/DeepWt_U6.hd5`
- `models/esp_rnn_model.hd5`
- `models/hf_rnn_model.hd5`

## What You Must Return
I need a complete implementation dossier that we can drop into our repo and execute.

## Deliverables Required

### A. Exact Implementation Strategy
Give a short but concrete implementation plan for:
- on-target gap closure
- off-target fallback or unblock strategy
- crispAI parity reconciliation
- integrated benchmark scaffolding

This section must say exactly what we should implement first, second, and third.

### B. Exact File Tree
Return the exact file tree we should add or modify, for example:
- `models/...`
- `scripts/...`
- `configs/...`
- `data/.../manifests/...`
- `cluster/...`
- `reports/...`

Mark each file as one of:
- `new`
- `modify`
- `generated`

### C. Full Code For New Files
For every new file you propose, provide full runnable code.
No placeholders.
No pseudocode.
No “implement this yourself.”

Priority files to include if they are needed:
- a strict CRISPR-FMC-protocol-aligned trainer or wrapper
- an HNN transfer/HL60-focused trainer if needed
- a crispAI parity evaluator that tests candidate target columns / aggregations / checkpoints
- a DeepHF artifact loader / validator if needed
- an off-target fallback frame builder if primary CCLMoff remains blocked
- scoreboard update/render logic if metric tracking changes
- cluster submission wrappers if new jobs are needed

### D. Exact Patches For Existing Files
For each existing file that must change, provide either:
- a unified diff patch, or
- full replacement content

Each patch must be tied to a clear reason.

### E. Exact Manifests And Split Definitions
Provide the exact JSON/CSV/YAML content for any needed manifests.
This includes:
- on-target fold definitions if you want them pinned differently
- WT->HL60 transfer manifest if needed
- crispAI exact parity manifest if it needs extra fields
- off-target fallback frame manifests
- integrated benchmark candidate manifest

Every manifest must specify:
- source files
- split logic
- target column
- metric
- claim-validity status

### F. Exact Command Sequence
Return the exact commands we should run, in order, for:
1. downloading assets
2. verifying artifacts and hashes
3. staging data
4. preprocessing
5. training
6. evaluation
7. summary generation
8. scoreboard regeneration

Commands must be copy-paste runnable.
Use exact paths where possible.

### G. Exact Cluster Execution Plan
We use clusters and want parallel execution.
Return exact cluster-ready commands or job scripts for:
- the next on-target near-gap sweep
- the HL60-focused run
- the crispAI parity diagnosis sweep
- the off-target fallback benchmark if needed

For each job specify:
- GPU/CPU requirements
- walltime
- batch size or effective batch size
- kill rules
- expected outputs

### H. Exact Experiment Matrix
Return the next 10 highest-value runs only.
For each run provide:
- experiment id
- purpose
- exact model
- exact config
- exact target metric(s)
- exact expected benefit
- exact promotion rule
- exact kill rule

Do not include speculative low-value runs.

### I. Exact Claim Logic
Provide the exact logic we should encode in reports for these cases:
- if on-target flips but off-target primary is still blocked
- if crispAI exact-bundle remains below 0.5114
- if off-target must use fallback frames only
- if DeepHF-aligned runs differ from CRISPR_HNN/CRISPR-FMC CSV runs

This should be written as exact reporting rules or conditional text blocks.

### J. Exact Final Recommendation
End with:
- the single next code file we should implement first
- the single next run we should launch first
- the single biggest remaining blocker after that
- the single most realistic path to a claim-valid “we outperform SOTA” result

## Constraints
- Use exact dates.
- Use exact metrics.
- Use exact filenames where possible.
- If something is not provable, label it `NOT YET PROVEN`.
- Separate claim-valid from proxy or fallback at all times.
- Prefer the smallest complete runnable implementation over a larger vague one.
- Assume we want to use clusters rather than local execution whenever reasonable.
- Do not omit DeepHF.
- Do not omit HL60.
- Do not omit the crispAI parity divergence.
- Do not omit the CCLMoff provenance blocker.

## Output Format
Return in this exact order:
1. Implementation Strategy
2. File Tree
3. New File Code
4. Existing File Patches
5. Manifests
6. Command Sequence
7. Cluster Plan
8. Experiment Matrix
9. Claim Logic
10. Final Recommendation
