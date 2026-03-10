# Follow-Up Deep Research Prompt 2 (2026-03-09)

Use this prompt with a deep-research system. This is a second follow-up pass, narrower than the previous ones. The purpose is to resolve the remaining artifact-level uncertainties that still block a claim-valid "beat latest SOTA" conclusion.

Current repo context:
- Strict scoreboard: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-09.json`
- Latest scoreboard markdown: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-09.md`
- Execution status: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/PUBLIC_EXECUTION_STATUS_2026-03-05.json`
- Upstream repro status: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_UPSTREAM_REPRO_STATUS_2026-03-05.json`

Current strict on-target gaps:
- mean9 SCC: 0.6743715317 vs 0.716 (gap -0.0416284683)
- WT SCC: 0.8453855978 vs 0.861 (gap -0.0156144022)
- ESP SCC: 0.8361978320 vs 0.851 (gap -0.0148021680)
- HF SCC: 0.8383885437 vs 0.865 (gap -0.0266114563)
- Sniper-Cas9 SCC: 0.9274682255 vs 0.935 (gap -0.0075317745)
- HL60 SCC: 0.3130585941 vs 0.402 (gap -0.0889414059)
- WT->HL60 SCC: 0.4653505492 vs 0.468 (gap -0.0026494508)

Current strict off-target primary status:
- all primary claim-valid rows still unavailable
- blocker: CCLMoff method provenance / frame validity

Current uncertainty status:
- best tracked Spearman: 0.5702236369 vs 0.5114
- numerically above target, but still non-claim-valid due parity mismatch

## Hard-Fail Instruction

Do not return a partial answer. I need one consolidated response that fully covers:
- exact CCLMoff public artifact inspection results
- exact crispAI Zenodo bundle inspection results
- exact CRISPR_HNN / CRISPR-FMC repo-content parity results
- exact DeepHF dataset/protocol alignment and its relevance to current targets
- exact claim-validity verdicts
- exact next acquisition and execution steps

Every material factual claim must include a direct URL.
If you make an inference, label it explicitly as an inference.
If an artifact must be downloaded and inspected, tell me exactly which file, what you expect to find, and how that finding would change the claim-validity status.

## Prompt

```text
You are doing a second follow-up deep-research pass for a CRISPR benchmark project. This is not a broad review. Resolve the remaining artifact-level uncertainties that prevent a claim-valid "beat latest SOTA" conclusion.

You must answer the following questions precisely and with direct URLs.

### Section 1: CCLMoff artifact-level inspection

Known facts:
- CCLMoff paper: publicly available
- CCLMoff repo: publicly available
- CCLMoff Figshare DOI: 10.6084/m9.figshare.27080566.v2
- Publicly visible files include:
  - `09212024_CCLMoff_dataset.csv`
  - `CCLMoff_V1.ckpt`
- The dataset CSV contains a large blank `Method` bucket
- Observed blank bucket rows: 3,282,935
- Paper Table 1 reports:
  - CIRCLE-seq rows: 1,683,395
  - bulged GUIDE-seq rows: 1,599,541
  - sum: 3,282,936
- Difference vs blank bucket: 1 row

Your tasks:
1. Inspect all publicly visible CCLMoff artifacts:
   - paper
   - supplement
   - Figshare article metadata
   - every Figshare file listed publicly
   - GitHub repo tree
   - Docker image documentation and, if publicly inspectable, image contents / manifest
2. Determine whether any public artifact contains:
   - per-row method annotation
   - per-assay split manifests
   - preprocessing outputs with resolved methods
   - a deterministic reconstruction key for the blank `Method` bucket
3. If the Docker image is a plausible source of hidden preprocessing artifacts, state exactly:
   - what files or directories we should inspect inside the container
   - what findings would count as sufficient provenance
   - what findings would still be insufficient for claim validity
4. Return a definitive verdict:
   - `claim-valid reconstructable from public artifacts: yes/no`
5. If yes, provide exact reconstruction steps.
6. If no, provide the exact minimal missing artifact needed from authors.
7. Also provide the strongest defensible fallback protocol built from original public assay datasets.

Output must include:
- exact file manifest
- exact verdict
- exact fallback hierarchy

### Section 2: crispAI artifact-level parity inspection

Known facts:
- crispAI repo exists publicly
- repo includes environment instructions and R/NuPoP dependencies
- Zenodo 12609337 is the data artifact
- Zenodo 13335960 is the model artifact
- claim-valid parity requires exact processed table, exact split, exact target column, exact metric path

Your tasks:
1. Inspect the crispAI repo and the Zenodo records as far as public metadata allows.
2. Determine whether the Zenodo records expose or imply:
   - exact filenames
   - processed tables
   - split indices
   - test-set definitions
   - target columns
3. Identify the exact evaluation path that produces the published 0.5114 Spearman.
4. Determine whether the repo alone is sufficient for parity, or whether parity depends on hidden bundle internals.
5. Return an explicit verdict:
   - `claim-valid crispAI parity reproducible from public artifacts: yes/no/not yet proven`
6. If it is reproducible, provide the exact file list and exact commands.
7. If it is not yet proven, provide the exact artifact-inspection checklist we must execute after download.

Output must include:
- exact Zenodo URLs
- exact file expectations
- exact parity checklist
- exact claim-validity verdict

### Section 3: CRISPR_HNN and CRISPR-FMC repo-content parity

Known facts:
- Public GitHub repos now appear to exist for both CRISPR_HNN and CRISPR-FMC
- Their existence does not prove paper parity
- Current on-target targets align with CRISPR-FMC-reported values

Your tasks:
1. Inspect the public CRISPR_HNN and CRISPR-FMC repos carefully.
2. Determine whether each repo contains:
   - training code
   - preprocessing code
   - exact benchmark split logic
   - checkpoints or releases
   - requirements / environment files
   - dataset manifests or links
   - README statements that tie code to the paper results
3. For each repo, return a paper-parity verdict:
   - `paper-parity likely`
   - `partial`
   - `incomplete`
   - `unclear`
4. Extract the exact paper protocol details that matter for claim-valid comparison, especially for CRISPR-FMC:
   - folds
   - validation strategy
   - loss
   - optimizer
   - learning rate
   - seed
   - batch size
   - early stopping
5. Determine whether the current public repo contents are enough to reproduce the paper’s reported targets directly, or whether paper-only reconstruction is still necessary.
6. If checkpoints exist, give the exact paths and usage instructions.
7. If the repos are incomplete, give the exact missing pieces.

Output must include:
- exact repo URLs
- exact parity verdict per repo
- exact missing pieces per repo
- exact protocol details we should mirror in our own runs

### Section 4: DeepHF dataset and protocol alignment

Do not forget DeepHF. This section is mandatory.

Known facts:
- DeepHF is foundational for the WT / ESP / HF large-scale on-target benchmarks
- Those datasets anchor several of our strict thresholds
- Our current scoreboard still misses WT, ESP, and HF

Your tasks:
1. Inspect the DeepHF paper, repo, and dataset artifacts.
2. Determine the exact public dataset files for:
   - WT
   - ESP
   - HF
3. Determine the exact sample counts, file names, schema, and preprocessing format.
4. Determine whether the DeepHF public files are raw, preprocessed, encoded, or directly benchmark-ready.
5. Extract the exact protocol details used in DeepHF and how later works reuse or transform those datasets.
6. Determine whether our current targets for WT/ESP/HF correspond directly to:
   - DeepHF original results
   - later CRISPR_HNN / CRISPR-FMC re-benchmarking on the same public files
7. Provide an exact alignment note:
   - what constitutes a truly comparable WT / ESP / HF evaluation
   - what common preprocessing mistakes or normalization mismatches would invalidate comparison
8. Provide exact URLs and exact file names for the canonical DeepHF data artifacts.
9. If there are multiple public mirrors or reprocessed versions, rank them by claim-validity usefulness.

Output must include:
- exact DeepHF file manifest
- exact protocol alignment note
- exact comparability caveats

### Section 5: On-target near-gap closure research only

Do not give another broad on-target survey. Focus only on the smallest remaining gaps.

Current nearest gaps:
- WT->HL60: -0.0026494508
- Sniper-Cas9: -0.0075317745
- ESP: -0.0148021680
- WT: -0.0156144022
- HF: -0.0266114563
- mean9: -0.0416284683
- HL60 remains the hardest at -0.0889414059

Your tasks:
1. Based only on source-backed facts and tightly reasoned engineering inference, identify the 10 highest-value interventions most likely to close these specific remaining gaps.
2. For each intervention, state:
   - which exact metric(s) it targets
   - why it should help
   - whether it is source-backed or inference
   - whether it should be a scout or a full run
3. Explicitly incorporate the now-visible CRISPR_HNN / CRISPR-FMC repos if their contents support the recommendation.
4. Explicitly incorporate DeepHF alignment if relevant to WT/ESP/HF closure.
5. Return a small, disciplined experiment matrix rather than a huge search space.

### Section 6: Final deliverables

Return one consolidated answer with these sections:
A. Executive summary
B. CCLMoff artifact verdict
C. crispAI artifact verdict
D. CRISPR_HNN / CRISPR-FMC repo parity verdict
E. DeepHF dataset / protocol verdict
F. On-target near-gap closure plan
G. Exact acquisition manifest
H. Exact missing-artifacts manifest
I. Exact next code / script additions we should make
J. Exact risks and unresolved blockers

Do not return vague advice. Return the exact artifact-level working we need.
```
