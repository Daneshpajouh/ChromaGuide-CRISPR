# SOTA Source Repro Log (2026-03-05)

## Scope
- Build a strict acquisition registry for external SOTA model sources.
- Clone/update all reachable repos into `data/public_benchmarks/sources`.
- Pin commit-level provenance and URL reachability into a machine-readable status artifact.

## Artifacts Added
- `sota_model_source_registry.json`
- `scripts/acquire_sota_model_sources.py`
- `SOTA_SOURCE_REPRO_STATUS_2026-03-05.json`
- `SOTA_SOURCE_SMOKE_REPRO_2026-03-05.json`
- `SOTA_SOURCE_RUNTIME_SMOKE_2026-03-05.json`
- `data/public_benchmarks/acquisition/sota_source_acquisition_status.json`

## Acquisition Result (latest run)
- `total_entries`: 16
- `repo_present`: 12
- `repo_updated`: 12
- `repo_missing_url`: 4
- `docs_links_total`: 21
- `docs_links_ok`: 21
- `docs_links_failed`: 0

## Blocked (no public repo URL in registry)
- `on_target.crnncrispr`
- `off_target.dnabert_epi`
- `off_target.crispr_ip`
- `off_target.crispr_dnt`

## Smoke Repro Check
- Performed Python parse smoke checks (entry scripts) on cloned repositories.
- Result: all targeted entry scripts compiled successfully in local environment (`all_ok = true`).
- Artifact: `SOTA_SOURCE_SMOKE_REPRO_2026-03-05.json`

## Runtime Repro Smoke
- `CRISPR-FMC` upstream public smoke run executed successfully (WT, 1 fold, 1 epoch, 512-row cap).
- `CRISPR_HNN` runtime currently blocked by missing `tensorflow` in benchmark venv.
- `CCLMoff` runtime currently blocked by missing `rnafm`.
- `crispAI` CLI is runnable, but `offt-score` smoke is blocked by missing `genomepy`.
- Artifact: `SOTA_SOURCE_RUNTIME_SMOKE_2026-03-05.json`

## Notes
- This run is source-control and provenance hardening, not a claim-valid benchmark win by itself.
- Claim-valid SOTA comparison still depends on matched benchmark-frame execution and threshold crossing under `public_claim_thresholds.json`.
