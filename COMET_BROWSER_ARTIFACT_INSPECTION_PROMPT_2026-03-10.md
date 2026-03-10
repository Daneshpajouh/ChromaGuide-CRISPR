# Comet Agentic Browser Prompt: Artifact Inspection, Header Extraction, and Google Doc Report

Use this prompt exactly as written.

## Hard-Fail Instruction
Do not return a partial result.
Do not stop after opening a repo.
Do not stop after finding links.
Do not stop after listing files.
Do not stop without checking actual file headers, schemas, or directory contents where accessible.

I need one consolidated output that includes:
- exact inspected URLs
- exact repo/file paths visited
- exact file lists discovered
- exact CSV/TSV header rows or table schemas where available
- exact checkpoint/model filenames where available
- exact commit hashes/releases/tags where available
- exact verdicts for claim-validity blockers
- exact missing artifacts still needed
- a Google Doc deliverable URL containing the final report

If the Google Doc URL is missing, the task is incomplete.

## Goal
Inspect the public artifacts around these CRISPR SOTA components and document the exact file-level evidence we still need for implementation and claim-valid benchmarking.

## Current Known State
We already know the following from local inspection and execution:

1. **CCLMoff primary off-target is claim-blocked**
- Public Figshare currently exposes only:
  - `09212024_CCLMoff_dataset.csv`
  - `CCLMoff_V1.ckpt`
- The blank `Method` bucket has `3,282,935` rows.
- That is consistent with a collapsed merge of CIRCLE-seq + bulged GUIDE-seq.
- Docker `/workspace` inspection did not reveal a hidden method map or split manifest.
- We still need to know whether any public artifact surface exposes:
  - a method/split map
  - extra manifests
  - hidden attachment files
  - paper supplement data tables

2. **crispAI is now a reproduction-divergence problem**
- Exact-bundle train/test CSVs and best checkpoint are staged locally.
- Exact-bundle rerun reproduced Spearman `0.4539840624` vs frozen target `0.5114`.
- We need exact file-level information about:
  - header rows / column names in the Zenodo data bundle
  - any evaluation notebooks/scripts in the public bundle
  - any mention of grouping, aggregation, target column, transform, or saved predictions

3. **CRISPR_HNN and CRISPR-FMC public repo parity is partial**
- Public repos exist.
- We need exact repo-tree evidence for:
  - training scripts
  - data loaders
  - fold-generation code
  - loss functions
  - optimizer settings
  - checkpoints/releases/tags/assets
  - dataset files and headers

4. **DeepHF is pinned locally but still needs browser-side documentation**
- We need exact public repo evidence for:
  - WT / ESP / HF `.pkl` files
  - `.hd5` models
  - exact public filenames
  - any docs about feature extraction / normalization / sequence format

## Required Targets To Inspect
You must inspect these public surfaces if accessible:

### Repos / sites
- `https://github.com/duwa2/CCLMoff`
- `https://github.com/xx0220/CRISPR_HNN`
- `https://github.com/xx0220/CRISPR-FMC`
- `https://github.com/izhangcd/DeepHF`
- `https://github.com/furkanozdenn/crispr-offtarget-uncertainty`
- any linked releases / tags / attachments / README-linked assets
- any linked Figshare / Zenodo / Docker Hub / supplement pages

### Artifact sources
- CCLMoff Figshare DOI / file page
- crispAI Zenodo records for data and model bundles
- Docker Hub or registry page for `weiandu/cclmoff:gpu`
- PMC/paper supplement pages if they expose file tables or downloadable supplements

## Exact Questions To Answer

### A. CCLMoff
1. On the public Figshare page, what exact files are listed?
2. Are there any additional attachments besides:
   - `09212024_CCLMoff_dataset.csv`
   - `CCLMoff_V1.ckpt`
3. Does the repo, README, releases, issues, wiki, or paper supplement expose:
   - method labels
   - split manifests
   - CIRCLE-only or GUIDE-only processed tables
   - LODO fold files
4. Does Docker Hub or container documentation expose any more artifact names than the GitHub repo does?
5. Final verdict: is there any public browser-visible evidence that unblocks claim-valid primary off-target reconstruction?

### B. crispAI
1. On Zenodo, what exact filenames are listed for:
   - the data bundle
   - the model/code bundle
2. If Zenodo/browser preview exposes filenames inside archives, list them.
3. On GitHub, what exact scripts appear relevant to:
   - test-set evaluation
   - prediction generation
   - grouping/aggregation
   - calibration/uncertainty scoring
4. If any CSV/TSV is browser-viewable, capture exact header rows.
5. Identify any explicit mention of:
   - target column name
   - grouping by sgRNA / guide
   - mean vs median prediction
   - Box-Cox / jitter / transforms
   - split indices or 70/20/10 rule
6. Final verdict: what exact public artifact or script path is most likely responsible for reproducing `0.5114`?

### C. CRISPR_HNN
1. What exact files are present in the repo root and key subfolders?
2. Are there releases, tags, or checkpoints?
3. What exact dataset files are present?
4. For any browser-viewable CSV/TSV, capture the exact header row.
5. Identify exact files for:
   - training
   - evaluation
   - data loading
   - fold generation / split generation
   - loss/optimizer definition
6. Final verdict: public train-only repo, or public parity-capable repo?

### D. CRISPR-FMC
1. What exact files are present in the repo root and key subfolders?
2. Are there releases, tags, or checkpoints?
3. What exact dataset files are present?
4. For any browser-viewable CSV/TSV, capture the exact header row.
5. Identify exact files for:
   - one-hot branch
   - RNA-FM embedding path
   - fusion / cross-attention
   - training loop
   - evaluation loop
   - split generation
   - optimizer/loss configuration
6. Final verdict: public train-only repo, or public parity-capable repo?

### E. DeepHF
1. What exact `.pkl` and `.hd5` files are present publicly?
2. What exact filenames are listed in `data/` and `models/`?
3. What exact docs mention:
   - sequence format
   - feature extraction
   - normalization
   - environment versions
4. Final verdict: what are the canonical public DeepHF artifacts we should cite and align to?

## Header Extraction Requirements
Whenever you find a browser-viewable CSV/TSV or rendered table, capture:
- exact header row
- exact source URL
- exact file path in repo/site

If the file is not directly viewable in browser but only downloadable, say:
- `download-only, header not browser-visible`

Do not guess headers.

## Required Deliverable Structure
Create a final report in a Google Doc with these sections:
1. Executive Summary
2. CCLMoff Artifact Inspection
3. crispAI Artifact Inspection
4. CRISPR_HNN Repo Inspection
5. CRISPR-FMC Repo Inspection
6. DeepHF Repo Inspection
7. Exact Header Rows and Schemas
8. Exact Missing Artifacts Still Needed
9. Claim-Validity Verdicts
10. Implementation-Relevant Next Steps

## Output Requirements
At the end, return:
- Google Doc URL
- a short summary table with columns:
  - Component
  - Browser-inspected evidence found
  - Headers found?
  - Checkpoints/releases found?
  - Claim-parity verdict
  - Biggest missing artifact

## Quality Bar
This is not a general summary task.
This is an artifact-inspection and documentation task.
Use the browser actively.
Open pages.
Expand repo trees.
Inspect file pages.
Look at releases/tags when present.
Record exact filenames and headers only when actually visible.
