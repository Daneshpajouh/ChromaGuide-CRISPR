# Targeted Deep Research Prompt: Solve the Remaining SOTA Gaps

Use this prompt exactly as written. This is not a broad literature review request. It is a blocker-resolution request.

## Hard-Fail Instruction
Do not return a partial result. If you only give high-level ideas, or only papers, or only architecture suggestions, the task is failed.

I need one consolidated output that contains:
- exact root-cause analysis for why we are still below the latest claim-valid SOTA
- exact public artifact parity verdicts
- exact file-level evidence
- exact reproduction diagnosis steps
- exact code-level modifications to make
- exact experiment matrix to run next
- exact benchmark/frame definitions
- exact pass/fail deltas versus the frozen targets
- exact commands/checkpoints/repos/assets to use
- exact stop/go decision rules

If any of those are missing, the run is incomplete.

## Context
We are trying to beat the latest CRISPR SOTA in a claim-valid way across:
- on-target
- off-target primary
- uncertainty/calibration
- integrated design

Current strict scoreboard:

| Metric | Best | Target | Gap (best-target) | Pass |
|---|---:|---:|---:|---|
| on_target mean9 SCC | 0.6743715317 | 0.716 | -0.0416284683 | No |
| WT SCC | 0.8453855978 | 0.861 | -0.0156144022 | No |
| ESP SCC | 0.8361978320 | 0.851 | -0.0148021680 | No |
| HF SCC | 0.8383885437 | 0.865 | -0.0266114563 | No |
| Sniper-Cas9 SCC | 0.9274682255 | 0.935 | -0.0075317745 | No |
| HL60 SCC | 0.3130585941 | 0.402 | -0.0889414059 | No |
| WT->HL60 SCC | 0.4653505492 | 0.468 | -0.0026494508 | No |
| crispAI exact-bundle test Spearman | 0.4539840624 | 0.5114 | -0.0574159376 | No |

Current known blockers:

1. **CCLMoff off-target primary claim-validity blocker**
- Public Figshare exposes only:
  - `09212024_CCLMoff_dataset.csv`
  - `CCLMoff_V1.ckpt`
- The blank `Method` bucket has `3,282,935` rows.
- That is consistent with a collapsed merge of CIRCLE-seq + bulged GUIDE-seq.
- Docker inspection did not reveal a hidden method map or split manifest.
- Therefore primary off-target frames are still blocked claim-validly.

2. **crispAI uncertainty reproduction divergence**
- Exact Zenodo code bundle is staged.
- Exact Zenodo train/test CHANGE-seq CSVs are staged.
- Exact best checkpoint is staged.
- Exact-bundle rerun fixed row-count drift to `168465`.
- But exact-bundle reproduced Spearman is still only `0.4539840624` vs frozen target `0.5114`.
- Therefore the uncertainty blocker is now a reproduction-divergence problem, not an asset-discovery problem.

3. **On-target is close on some rows, but still below**
- Closest gaps:
  - WT->HL60: `-0.0026494508`
  - Sniper-Cas9: `-0.0075317745`
  - ESP: `-0.0148021680`
  - WT: `-0.0156144022`
- Largest remaining problems:
  - HL60: `-0.0889414059`
  - mean9: `-0.0416284683`

4. **Public repo parity is partial, not complete**
- CRISPR_HNN repo exists with train code + datasets, but no public checkpoint/release assets pinned.
- CRISPR-FMC repo exists with train code + datasets, but no public checkpoint/release assets pinned.
- DeepHF canonical artifacts are pinned and available:
  - `wt_seq_data_array.pkl`
  - `esp_seq_data_array.pkl`
  - `hf_seq_data_array.pkl`
  - `.hd5` model files

## What I Want You To Solve
I want you to tell us exactly why we are still failing to outperform, and exactly what to do next to cross the line in a defensible, claim-valid way.

This must be a **solution dossier**, not a paper summary.

## Deliverables Required

### A. Root-Cause Table
For each blocked area:
- on-target
- off-target primary
- uncertainty
- integrated design

Provide:
- exact failure mode
- whether it is a modeling issue, data issue, protocol issue, provenance issue, or execution issue
- confidence level
- direct evidence
- exact fix priority

### B. On-Target Root-Cause Diagnosis
Use the exact current gaps above and tell me:

1. Which of these are most likely due to protocol mismatch versus model weakness:
- mean9
- WT
- ESP
- HF
- Sniper-Cas9
- HL60
- WT->HL60

2. For CRISPR_HNN and CRISPR-FMC, identify:
- exact public code path for fold generation
- exact loss
- exact optimizer
- exact batch size
- exact seed
- exact validation strategy
- exact normalization policy
- exact sequence length / input encoding
- whether checkpoints are public or only train scripts are public

3. Tell me exactly which small changes are most likely to flip the closest rows from fail to pass:
- WT->HL60
- Sniper-Cas9
- ESP
- WT

4. Tell me exactly what is needed to move:
- HL60
- mean9

5. Return an ordered experiment plan with:
- exact architecture
- exact hyperparameters
- exact training schedule
- exact transfer strategy
- exact promotion rule
- exact kill rule

### C. Off-Target Primary Claim-Valid Recovery
This section must answer a hard yes/no:

**Can CCLMoff primary claim-valid frames be reconstructed from public artifacts alone?**

If `no`, say `NO` clearly and explain exactly why.
If `yes`, provide exact files, commands, and mapping logic.

You must inspect all plausible public surfaces conceptually and tell us the strongest realistic path:
- Figshare
- repo
- Docker/container
- paper supplement
- original assay papers

Then provide:

1. exact minimum artifact needed to unblock claim-validity
2. exact near-claim-valid fallback benchmark we should run instead if primary CCLMoff parity remains impossible
3. exact dataset/frame construction recipe for that fallback
4. exact claim wording we can safely make if we use the fallback instead of true CCLMoff primary parity

### D. crispAI Uncertainty Reproduction Diagnosis
We now have:
- exact train CSV
- exact test CSV
- exact checkpoint
- exact supplementary scripts
- exact-bundle rerun with row count `168465`
- reproduced Spearman `0.4539840624`
- frozen target `0.5114`

I need you to diagnose the likely remaining causes of this gap.

You must provide:

1. a ranked list of likely causes for the remaining `-0.0574159376` gap
2. whether the published `0.5114` is likely:
   - mean prediction
   - median prediction
   - another saved array
   - another transform path
   - another target column
   - another split regime
3. whether the random jitter / Box-Cox fitting path could explain part of the gap
4. whether the published number may come from:
   - a saved array in the bundle
   - another script in the bundle
   - a different checkpoint than `epoch:19-best_valid_loss:0.270.pt`
   - a hidden validation/test selection rule not yet applied

Then provide the exact next 5 inspection or execution steps to close the parity gap.

### E. DeepHF Canonical Alignment
Do not ignore DeepHF.

I need:
- exact canonical DeepHF artifact inventory
- exact encoding assumptions
- exact normalization assumptions
- exact train/test or CV protocol assumptions
- exact comparability note between DeepHF repo data and CRISPR_HNN/CRISPR-FMC bundled CSVs
- exact risks of using reprocessed CSVs vs original `.pkl` arrays

### F. Exact Experiment Matrix To Beat SOTA
Return the next **15 highest-value experiments only**.

For each experiment provide:
- experiment name
- target metric(s)
- expected gain
- evidence basis
- exact model family
- exact data
- exact split
- exact hyperparameters
- exact compute level
- exact failure risk
- exact stop/promotion rule

Prioritize only experiments that can realistically move us from the current rows to pass/fail flips.
Do not waste slots on broad speculative ideas.

### G. Exact Commands / Assets / Repos
Return:
- exact clone commands
- exact download commands
- exact commit hashes if known
- exact file paths expected after download
- exact checkpoint file names
- exact scripts we should write next if missing

### H. Exact Safe Claim Language
Give claim language for four cases:
- if on-target passes but off-target primary remains blocked
- if uncertainty exact-bundle still misses published target
- if off-target must use fallback near-claim-valid frames
- if we outperform all available public baselines but not frozen paper parity

### I. Exact Decision Tree
Produce a strict decision tree:

1. If crispAI exact parity is solved, do X
2. If crispAI exact parity still misses, do Y
3. If CCLMoff primary remains impossible, do Z
4. If on-target closest rows flip but mean9 still fails, do W
5. If no promoted run improves after N attempts, pivot to Q

### J. Final Priority Verdict
End with:
- the single highest-value next step
- the single biggest scientific blocker
- the single biggest engineering blocker
- the single best path to a defensible “we outperform SOTA” claim

## Constraints
- Use exact dates.
- Use exact metrics.
- Use exact file names where possible.
- If something is not provable, say so explicitly.
- Do not give generic suggestions.
- Do not tell me to “explore more” without specifying exact actions.
- Do not return partial work.

## Desired Output Format
Return the answer in sections `A` through `J` exactly.
Use compact tables where useful.
Use exact numbers and exact deltas.
Mark each verdict as one of:
- `YES`
- `NO`
- `UNCLEAR`
- `NOT YET PROVEN`

