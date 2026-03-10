# Follow-Up Deep Research Prompt (2026-03-09)

Use this prompt with a strong deep-research system. This is not a broad literature review request. It is a narrow claim-hardening follow-up intended to resolve the remaining blockers to a real, claim-valid "beat latest SOTA" result.

Current repo state:
- Repo: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR`
- Strict scoreboard: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-09.json`
- Latest execution status: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/PUBLIC_EXECUTION_STATUS_2026-03-05.json`
- Upstream repro status: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_UPSTREAM_REPRO_STATUS_2026-03-05.json`

Current exact strict gaps:

| Metric | Best | Target | Gap (best-target) | Pass |
|---|---:|---:|---:|---|
| on_target mean9 SCC | 0.6743715317 | 0.716 | -0.0416284683 | No |
| WT SCC | 0.8453855978 | 0.861 | -0.0156144022 | No |
| ESP SCC | 0.8361978320 | 0.851 | -0.0148021680 | No |
| HF SCC | 0.8383885437 | 0.865 | -0.0266114563 | No |
| Sniper-Cas9 SCC | 0.9274682255 | 0.935 | -0.0075317745 | No |
| HL60 SCC | 0.3130585941 | 0.402 | -0.0889414059 | No |
| WT->HL60 SCC | 0.4653505492 | 0.468 | -0.0026494508 | No |
| uncertainty CHANGE-seq test Spearman | 0.5702236369 | 0.5114 | +0.0588236369 | Yes, but not claim-valid |

Current exact off-target claim-valid blocker:
- Primary off-target metrics are still unavailable.
- The public CCLMoff Figshare CSV has a huge blank `Method` bucket.
- Observed blank bucket rows: `3,282,935`
- CCLMoff paper Table 1 reported:
  - CIRCLE-seq rows: `1,683,395`
  - bulged GUIDE-seq rows: `1,599,541`
  - sum: `3,282,936`
- Difference vs blank bucket: `1`
- This strongly suggests the blank bucket is effectively a mixed collapsed `CIRCLE-seq + bulged GUIDE-seq` bucket, not a clean method-specific frame.
- Therefore current public assets do **not** support claim-valid primary off-target benchmarking unless a source-verified method mapping exists somewhere else.

Current exact uncertainty claim-valid blocker:
- Our current uncertainty result is on a proxy CHANGE-style frame.
- We need the exact crispAI parity data artifact, exact split rule, exact target column, and exact evaluation code path.

Current exact upstream reproducibility blocker:
- CRISPR_HNN and CRISPR-FMC are reported as public/latest SOTA in recent papers, but their public code/checkpoint availability remains unresolved.
- We need a definitive answer on whether public repos/checkpoints/supplements exist, whether code was released under another name, or whether full reproduction must rely on paper-only reconstruction.

## Follow-Up Research Prompt

```text
You are doing a narrow follow-up deep research pass for a CRISPR benchmark project. Do not return a broad survey. Resolve the remaining claim-validity blockers with source-backed evidence, exact URLs, exact file names, and exact reconstruction steps.

You must answer these questions precisely:

1. Off-target primary frame validity
2. Uncertainty claim-valid parity
3. Upstream SOTA code/checkpoint availability
4. Exact surpass strategy for the remaining on-target gaps
5. Exact integrated benchmark design if we clear component benchmarks

Hard-fail instruction:
Do not return a partial result. If you only cover on-target or only off-target or only repo links, the task is failed. I need one consolidated answer that fully covers:
- off-target frame validity and method-resolution paths
- uncertainty parity artifact resolution
- upstream repo/checkpoint/supplement acquisition
- exact paper-to-code reconstruction requirements
- exact latest SOTA verification
- exact claim-valid benchmark definitions
- exact recommendations to surpass all current bests
- exact missing public assets
- exact next experiments and next code files

Use only current, source-backed information. Every non-trivial factual claim must include a direct URL.
When making an inference, label it explicitly as an inference.
If public assets are insufficient, say so clearly and explain why.
If a repo is missing, prove that you searched and state what alternative artifact exists.

### Section 1: Off-target primary frame unblocker

We need a definitive answer on whether the CCLMoff primary claim-valid frames can be reconstructed from public assets.

Known facts:
- Figshare dataset DOI: 10.6084/m9.figshare.27080566.v2
- Files publicly visible there include:
  - `09212024_CCLMoff_dataset.csv`
  - `CCLMoff_V1.ckpt`
- The CSV has fields including `sgRNA_seq`, `off_seq`, `read`, `sgRNA_type`, `label`, `chr`, `Location`, `Direction`, `Length`, `Method`, `id`
- Large portions of `Method` are blank
- Blank bucket rows: 3,282,935
- CCLMoff paper-reported CIRCLE-seq rows: 1,683,395
- CCLMoff paper-reported bulged GUIDE-seq rows: 1,599,541
- Sum: 3,282,936
- Therefore the blank bucket appears to be a merged CIRCLE+GUIDE-bulge pool, not a clean method-specific frame

Your job:
1. Search all public artifacts around CCLMoff:
   - paper
   - supplementary materials
   - Figshare article and file metadata
   - GitHub repo
   - Docker image docs
   - any linked preprocessing scripts
   - issue tracker / discussions if public
2. Determine whether any public artifact contains the true per-row method annotation or a deterministic reconstruction path.
3. If yes:
   - give the exact URL
   - exact file name(s)
   - exact schema
   - exact steps to resolve the blank bucket
   - exact steps to build these claim-valid frames:
     - CIRCLE-seq CV
     - CIRCLE->GUIDE transfer
     - DIG-seq LODO
     - DISCOVER-seq+ LODO
4. If no:
   - state clearly that CCLMoff primary claim-valid frames cannot be reconstructed from public assets alone
   - provide the most defensible fallback hierarchy:
     - claim-valid if authors provide X
     - near-claim-valid if we independently reconstruct Y
     - proxy-only if we use Z
5. For the independent reconstruction fallback, provide a fully specified plan:
   - exact original assay papers and datasets to use
   - exact download URLs
   - exact negative-construction rules
   - exact mismatch cap
   - exact bulge policy
   - exact genome build
   - exact filtering rules
   - exact balancing/downsampling
   - exact split/fold rules
6. Also evaluate whether newer public off-target models after CCLMoff change the frozen SOTA picture or benchmark design. Include at minimum:
   - CRISPR-MFH
   - CRISMER
   - CrisprDNT
   - CRISPR-MCA
   - DNABERT-Epi
   - crispAI
   - any public 2025–2026 model with stronger numbers or better public reproducibility

Output for this section must include:
- a yes/no answer on claim-valid reconstructability from public artifacts
- a file manifest
- a protocol manifest
- an author-contact escalation package if needed

### Section 2: Uncertainty claim-valid parity resolution

We need to convert uncertainty from a proxy result into a claim-valid result.

Known facts:
- Current best tracked uncertainty number: Spearman 0.5702236369
- Frozen target: 0.5114
- The current number is not claim-valid because it is not on the exact crispAI parity frame
- crispAI repo: `furkanozdenn/crispr-offtarget-uncertainty`
- Reported Zenodo records mentioned so far:
  - 12609337 (data)
  - 13335960 (model)

Your job:
1. Identify the exact crispAI parity artifact:
   - exact Zenodo record URL
   - exact file names
   - exact processed table(s)
   - exact target column(s)
   - exact split rule (70/20/10 or otherwise)
   - exact random seed if specified
   - exact metric definition used for the reported 0.5114
2. Determine whether the split indices are included publicly.
3. Determine whether crispAI remains the latest public uncertainty/calibration SOTA as of today, or whether any newer public method supersedes it.
4. Return a full reproduction recipe:
   - exact environment
   - exact preprocessing steps
   - exact dependencies, including R/NuPoP or any physical-feature generation
   - exact evaluation commands if available
5. Tell us whether our current proxy frame could be transformed into claim-valid parity, and if not, what is missing.
6. Provide a surpass plan for uncertainty:
   - source-backed or tightly inferred changes only
   - exact candidate losses, calibration methods, or selection metrics
   - exact claim-valid metric(s) we should report beyond raw Spearman if appropriate

Output for this section must include:
- exact download list
- exact reproduction steps
- exact claim-valid parity test definition
- exact surpass plan

### Section 3: CRISPR_HNN and CRISPR-FMC acquisition and reproducibility resolution

We need a definitive source-backed answer on the current public status of CRISPR_HNN and CRISPR-FMC.

Known facts:
- They are central to the current on-target SOTA discussion
- We have reconstructed and run upstream-style approximations
- But public repo/checkpoint availability remains unresolved

Your job:
1. Search rigorously for public repos, mirrors, forks, supplemental archives, model checkpoints, institutional pages, and code releases for:
   - CRISPR_HNN
   - CRISPR-FMC
2. If found, return:
   - exact repo URL
   - exact branch/tag/release
   - exact checkpoint files
   - exact dependency stack
   - exact dataset/preprocessing instructions
3. If not found, provide a source-backed proof trail that they are not publicly available:
   - paper links
   - supplementary links
   - code availability statements in the papers
   - any broken or absent repo links
4. For each model, provide a full reconstruction table:
   - architecture diagram in words
   - exact input representations
   - exact modules/layers
   - exact losses
   - exact optimizer/schedule
   - exact training protocol
   - exact external pretrained dependencies
   - exact hardware profile
   - exact unclear/missing parts
5. Tell us which of their claimed gains are most likely due to:
   - architecture
   - pretraining
   - multi-task training
   - preprocessing choices
   - split differences
6. Provide a ranked list of what we should borrow or not borrow from each model in order to surpass them.

Output must separate:
- directly source-supported facts
- engineering inference from source-supported facts
- unresolved unknowns

### Section 4: On-target gap-closure follow-up

We do not need another generic on-target survey. We need the highest-value interventions for the remaining exact gaps.

Current exact gaps:
- mean9: -0.0416284683
- WT: -0.0156144022
- ESP: -0.0148021680
- HF: -0.0266114563
- Sniper: -0.0075317745
- HL60: -0.0889414059
- WT->HL60: -0.0026494508

Recent facts from our own runs:
- tuned HNN v3 improved mean9 to 0.6743715317
- tuned HNN v2 improved Sniper-Cas9 to 0.9274682255
- transfer v2 did not beat the standing WT->HL60 best
- WT remains held by a non-HNN baseline at 0.8453855978

Your job:
1. Give a ranked list of the 15 most evidence-backed interventions most likely to close these exact gaps.
2. For each intervention, state:
   - which metric(s) it targets
   - why it should help
   - whether the support is direct evidence or inference
   - whether it should be tried as a scout or a full run
3. Focus especially on:
   - HL60 / WT->HL60 transfer
   - WT and Sniper, which are closest to passing
   - mean9, which is still below threshold by 0.0416
4. Explicitly analyze whether we should prioritize:
   - multi-task across all 9 datasets
   - domain-adaptive heads
   - ranking/Spearman-aware loss
   - RNA-FM or DNABERT-2 embeddings
   - Mamba/state-space blocks
   - ensembling
   - curriculum learning
   - cell-type epigenomic features for HL60
5. Give a constrained Optuna/NAS search space that is justified and not wasteful.
6. Give a promotion policy and a stopping policy.

Output must include a concrete experiment matrix ready for cluster submission.

### Section 5: Integrated benchmark follow-up

We need a more rigorous answer on integrated design claims.

Your job:
1. Determine whether any public 2025–2026 work introduces a benchmarkable integrated CRISPR guide-selection metric suite.
2. If yes, provide exact protocol and assets.
3. If no, design the strongest defensible benchmark buildable now from public assets, including:
   - ranking metric(s)
   - uncertainty-aware metric(s)
   - off-target filtering integration
   - top-k selection quality
   - exact datasets
   - exact evaluation protocol
4. Tell us how to make an integrated claim without overclaiming beyond component metrics.

### Section 6: Deliverables

Return one consolidated answer with these sections:
A. Executive summary
B. Off-target primary frame verdict
C. Uncertainty parity verdict
D. CRISPR_HNN / CRISPR-FMC acquisition verdict
E. On-target gap-closure experiment plan
F. Integrated benchmark plan
G. Exact acquisition manifest (URLs, file names, repos, checkpoints)
H. Exact missing-assets manifest
I. Exact code/files/scripts we should add to our repo next
J. Exact risks and unresolved blockers

Do not return generic advice. Return the exact working we need to implement.
```

## Notes for the external researcher

What would be most valuable to resolve:
1. A public artifact that truly resolves the CCLMoff blank `Method` bucket
2. The exact crispAI parity data/split artifact
3. A definitive acquisition path or non-availability proof for CRISPR_HNN and CRISPR-FMC
4. A sharper, source-backed experiment plan specifically for the remaining on-target gaps
