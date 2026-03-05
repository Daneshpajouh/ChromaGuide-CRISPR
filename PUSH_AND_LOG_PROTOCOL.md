# Push and Logging Protocol

This repository now uses the following operating rule for benchmark work:

1. Every experiment wave must be logged in a dated run-log file (`RUN_LOG_YYYY-MM-DD_*.md`) with:
- cluster host
- submitted job IDs
- script/config used
- output artifact paths
- key metrics and threshold gaps
- blockers/failures

2. Any code/script change that affects acquisition, training, evaluation, or orchestration must be committed and pushed to `origin` immediately after validation.

3. For cluster submissions, log both:
- smoke status per cluster
- full-scale submission status per cluster

4. Claim tracking must always reference:
- `public_claim_thresholds.json`
- `benchmark_protocol_matrix.csv`

5. Off-target results must explicitly state whether they are:
- single split only
- full manifest sweep aggregate
- claim-valid frame match

6. If a run is blocked (scheduler/plugin/data provenance), the blocker must be recorded in the same-day run log and in status JSON updates.
