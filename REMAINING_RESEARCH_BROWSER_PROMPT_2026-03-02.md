# Remaining Research Browser Prompt (2026-03-02)

Use this prompt with the browser agent to close only the remaining research gaps. This is intentionally narrower than the full omnibus SOTA audit prompt.

```text
You are acting as a scientific benchmark extraction agent with full browser access and institutional journal access.

Your task is to complete only the remaining unresolved research gaps for a CRISPR benchmark dossier. The broad benchmark landscape is already known. Do not repeat the whole audit. Focus only on exact missing values, unresolved comparability details, and under-audited benchmark areas.

Scope

The current benchmark picture is already sufficiently established for:
- on-target single-model leadership (CRISPR_HNN as current public leader under the 9-dataset DeepHF-style regime),
- off-target public leadership (CCLMoff as current main public leader for cross-dataset generalization).

You must now focus only on the remaining gaps that block a final, fully defensible “we beat SOTA” claim.

Your tasks

1. Resolve exact missing values from papers and supplements

Priority papers:
- CRISPR_HNN (full text + supplement)
- DNABERT-Epi (main paper + supplementary tables)
- CRISPR-FMC (figure/table exact values, especially any values not explicitly extracted in prior summary)
- DeepMEns (exact per-dataset and independent-test values)
- DeepCRISTL / EXPosition / CRISPRon ensemble (exact benchmark values where directly relevant)

For each paper:
- retrieve all exact numeric benchmark values that were previously missing, approximate, figure-only, or abstract-only
- prioritize tables and supplementary XLSX/PDF files
- if extracting from figures, label each value as:
  - exact from annotation
  - or estimated from plot
- record:
  - exact source URL
  - DOI / PMID / PMCID if available
  - section / table / figure / supplement location

2. Complete calibration / uncertainty benchmarking

You must perform a focused audit of calibration / uncertainty papers, including:
- crispAI
- CRISPRon uncertainty-aware ensemble
- DeepEmbCas9 / DeepEnsEmbCas9
- any directly comparable 2024–2026 papers that report calibration metrics for CRISPR prediction

Extract exact metrics where available:
- ECE
- Brier score
- NLL
- empirical coverage
- interval width
- MAE / RMSE if tied to uncertainty reporting
- uncertainty decomposition (aleatoric / epistemic)

If a paper reports uncertainty qualitatively but not with standard calibration metrics, say that explicitly.

3. Complete integrated sgRNA design benchmark research

You must perform a focused audit of integrated sgRNA design tools and benchmark practices, including:
- CRISPick
- CHOPCHOP
- CRISPR-GPT
- EXPosition
- any 2024–2026 papers that define end-to-end sgRNA design or ranking utility

For each:
- identify whether it is a prediction model, a design system, or a ranking framework
- identify the exact utility definition used
- extract any benchmark metrics such as:
  - NDCG@k
  - Top-k hit rate
  - precision@k
  - ranking uplift
  - task success rate
- determine whether it is directly comparable to model-centric prediction papers

4. Build a negative-construction comparability audit for off-target work

This is a priority.

For all major off-target benchmark papers (especially CCLMoff, DNABERT-Epi, CRISPR-MCA, CRISPR-Net family, CRISPR-DNT, CRISPR-IP, MOFF):
- extract the exact negative construction pipeline
- record:
  - tool used (e.g. Cas-OFFinder)
  - max mismatches allowed
  - bulge handling
  - genome build
  - filtering rules
  - ratio of negatives to positives
- create a comparability table explaining which papers are directly comparable and which are not because of different negative construction

5. Complete transfer-learning / small-data comparability for on-target

For on-target benchmark papers:
- extract exact transfer-learning results for small datasets (HCT116 / HELA / HL60)
- distinguish:
  - plain 5-fold CV on the small dataset
  - pretrain on large datasets then fine-tune
  - zero-shot transfer
  - leave-one-cell-out

This section must explicitly state which results are fair comparisons to our future single-model claim if we also use pretraining.

Required outputs

Produce the following files:

1. `remaining_research_gap_closure.md`
- concise but exact
- must contain:
  - exact missing values found
  - what remains unavailable
  - calibration benchmark state
  - integrated-design benchmark state
  - off-target negative-construction comparability table
  - small-data / transfer-learning comparability table

2. `remaining_exact_values.csv`
- one row per extracted exact numeric value
- columns:
  - paper
  - year
  - task
  - model
  - dataset
  - protocol
  - metric
  - value
  - source_location
  - extraction_type
  - source_url

3. `offtarget_negative_construction_audit.csv`
- one row per paper
- columns:
  - paper
  - negatives_source
  - generator
  - max_mismatches
  - bulges_allowed
  - genome_build
  - filtering_notes
  - directly_comparable_to_cclmoff
  - reason

4. `calibration_benchmark_matrix.csv`

5. `integrated_design_benchmark_matrix.csv`

6. `transfer_learning_small_data_matrix.csv`

Method rules

- Use primary sources only.
- Use institutional access where needed.
- Prefer exact tables and supplements over prose summaries.
- If a value cannot be recovered exactly, say so and do not invent it.
- Explicitly mark every result as:
  - directly comparable
  - partially comparable
  - not directly comparable

This is a gap-closure task, not a general literature overview. Stay focused on unresolved details that matter for a final defensible SOTA claim.
```
