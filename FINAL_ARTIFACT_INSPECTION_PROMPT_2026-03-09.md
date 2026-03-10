# Final Artifact-Inspection Prompt (2026-03-09)

Use this with a deep-research system only if it can inspect downloadable artifacts, container contents, Zenodo/Figshare bundles, and repo trees at file level.

This is the final useful research prompt before execution should dominate. Do not do another broad review. Do not summarize papers generally. Inspect artifacts and return hard yes/no answers with file-level evidence.

Current repo context:
- Strict scoreboard: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-09.json`
- Execution status: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/PUBLIC_EXECUTION_STATUS_2026-03-05.json`
- Upstream repro status: `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_UPSTREAM_REPRO_STATUS_2026-03-05.json`

Current exact strict gaps:
- on_target mean9 SCC: 0.6743715317 vs 0.716 (gap -0.0416284683)
- WT SCC: 0.8453855978 vs 0.861 (gap -0.0156144022)
- ESP SCC: 0.8361978320 vs 0.851 (gap -0.0148021680)
- HF SCC: 0.8383885437 vs 0.865 (gap -0.0266114563)
- Sniper-Cas9 SCC: 0.9274682255 vs 0.935 (gap -0.0075317745)
- HL60 SCC: 0.3130585941 vs 0.402 (gap -0.0889414059)
- WT->HL60 SCC: 0.4653505492 vs 0.468 (gap -0.0026494508)
- uncertainty CHANGE-seq test Spearman: 0.5702236369 vs 0.5114 (numerically above target, but not claim-valid)

Current exact claim-validity blockers:
1. CCLMoff primary off-target frames are blocked by unresolved per-row method provenance.
2. crispAI uncertainty parity is blocked by unknown exact processed table/split/target mapping inside Zenodo bundles.
3. CRISPR_HNN and CRISPR-FMC repo existence is known, but paper-parity of code/checkpoints/preprocessing is not yet proven.
4. DeepHF is foundational for WT/ESP/HF and must be pinned down exactly as the canonical public artifact.

## Hard-Fail Instruction

Do not return partial output.
If you do not inspect the relevant artifact surfaces directly, the task is failed.
I need one consolidated answer with hard yes/no verdicts backed by specific files, filenames, paths, URLs, hashes if available, and exact reasons.

No generic advice.
No broad literature overview.
No “likely” unless you clearly label it as an inference after exhausting visible evidence.

## Primary Goal

Return definitive artifact-level answers for these four questions:
1. Can CCLMoff primary off-target frames be made claim-valid from public artifacts alone?
2. Can crispAI’s 0.5114 uncertainty parity frame be reproduced from public artifacts alone?
3. Do the public CRISPR_HNN and CRISPR-FMC repos actually support paper-parity reproduction?
4. What are the exact canonical DeepHF WT/ESP/HF public artifacts and protocol assumptions we must align to?

## Prompt

```text
You are doing a final artifact-inspection research pass for a CRISPR benchmark project. Your job is not to read papers generally. Your job is to inspect artifact surfaces and return hard yes/no verdicts.

You must inspect, at file level where possible:
- GitHub repositories
- release assets
- supplementary archives
- Docker image metadata and, if possible, container file trees
- Figshare metadata and downloadable file manifests
- Zenodo metadata and downloadable file manifests
- dataset bundle internal filenames if accessible

### Section 1: CCLMoff final artifact verdict

Known facts:
- Figshare DOI: 10.6084/m9.figshare.27080566.v2
- Publicly visible files include `09212024_CCLMoff_dataset.csv` and `CCLMoff_V1.ckpt`
- The dataset CSV includes columns like `Method`, but a huge bucket is blank
- Observed blank bucket rows: 3,282,935
- Paper Table 1 reported:
  - CIRCLE-seq rows: 1,683,395
  - bulged GUIDE-seq rows: 1,599,541
  - sum: 3,282,936
- Difference vs blank bucket: 1 row
- Repo advertises Docker image `weiandu/cclmoff:gpu`

Your tasks:
1. Enumerate every publicly visible CCLMoff artifact:
   - repo files
   - release assets if any
   - Figshare files
   - supplement files
   - Docker image references
2. Inspect whether any artifact contains or implies:
   - per-row method labels
   - assay-specific split manifests
   - resolved training/evaluation tables
   - scripts that deterministically map rows to methods
3. If the Docker image can be inspected, search for:
   - `.csv`, `.tsv`, `.json`, `.pkl`, `.pt`, `.ckpt`, `.yaml` files
   - directories named by assay or benchmark split
   - any training/evaluation manifest
4. Return a final verdict:
   - `claim-valid primary off-target reconstructable from public artifacts alone: YES or NO`
5. If YES:
   - give exact files/paths/URLs and reconstruction steps
6. If NO:
   - state the exact missing artifact needed
   - state whether Docker inspection narrowed the gap or not
7. Return the strongest defensible fallback benchmark path built from original public assay datasets, with exact source URLs.

Output must include:
- exact artifact inventory
- exact verdict
- exact missing artifact
- exact fallback path

### Section 2: crispAI final artifact verdict

Known facts:
- public GitHub repo exists
- Zenodo records 12609337 and 13335960 are relevant
- current claim-valid block is exact processed table + split + target column + eval path for 0.5114

Your tasks:
1. Enumerate all publicly visible crispAI artifacts:
   - repo tree
   - env files
   - scripts
   - Zenodo records and file manifests
2. If Zenodo bundle contents can be enumerated, return exact filenames.
3. Determine whether the public artifacts are sufficient to reproduce:
   - exact processed data table
   - exact train/val/test split or exact deterministic split rule
   - exact target column used for reported test Spearman 0.5114
   - exact evaluation script path
4. Return a final verdict:
   - `claim-valid crispAI parity reproducible from public artifacts alone: YES / NO / NOT YET PROVEN`
5. If YES:
   - provide exact filenames, commands, and evaluation path
6. If NOT YET PROVEN or NO:
   - provide the exact artifact-inspection checklist we must run locally after download
   - specify which missing file or ambiguous element is blocking parity

Output must include:
- exact Zenodo artifact manifest
- exact repo script manifest
- exact parity verdict
- exact next local inspection steps

### Section 3: CRISPR_HNN and CRISPR-FMC repo parity verdict

Known facts:
- public GitHub repos now appear to exist
- repo existence alone is not enough
- CRISPR-FMC paper provides a detailed training/evaluation protocol

Your tasks:
1. Inspect both repo trees at file level.
2. Determine whether each repo contains:
   - training code
   - preprocessing code
   - exact dataset loading code
   - exact fold-generation logic or fixed splits
   - checkpoints / releases / weights
   - environment/requirements files
   - explicit README statements tying code to the paper
3. Return for each repo one verdict:
   - `paper-parity likely`
   - `partial`
   - `incomplete`
   - `unclear`
4. For CRISPR-FMC, extract the exact protocol details that should be mirrored in our own runs.
5. For CRISPR_HNN, determine whether the repo is sufficient to replace paper-only reconstruction.
6. Identify exact missing pieces if any.

Output must include:
- exact repo URLs
- exact file-level evidence
- exact verdict per repo
- exact missing pieces per repo

### Section 4: DeepHF final canonical-artifact verdict

This section is mandatory.

Your tasks:
1. Enumerate the exact canonical DeepHF public artifacts for WT / ESP / HF.
2. Return exact filenames, file formats, and whether they are raw, encoded, or benchmark-ready.
3. Inspect how the DeepHF repo and paper define:
   - sequence representation
   - normalization
   - train/test usage
   - model files
4. Determine what later papers (especially CRISPR_HNN / CRISPR-FMC / DeepCRISTL) appear to reuse from DeepHF.
5. Return a verdict:
   - what exact DeepHF artifact should be treated as canonical for claim-valid WT/ESP/HF alignment
   - what preprocessing mismatch would invalidate comparison
   - whether public reprocessed mirrors are acceptable or only secondary convenience copies

Output must include:
- exact file manifest
- exact canonical-artifact verdict
- exact comparability note

### Section 5: Final deliverables

Return one consolidated answer with these sections:
A. Executive summary
B. CCLMoff final artifact verdict
C. crispAI final artifact verdict
D. CRISPR_HNN repo parity verdict
E. CRISPR-FMC repo parity verdict
F. DeepHF canonical-artifact verdict
G. Exact artifact acquisition manifest
H. Exact missing-artifact manifest
I. Exact local inspection checklist we should run next
J. Exact remaining unresolved blockers after this pass

Do not return generic recommendations. Return file-level evidence and final verdicts.
```
