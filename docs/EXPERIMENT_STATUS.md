# Experiment Deployment Status — Session 5

## Date: February 26, 2026, 02:00-08:00 UTC (PST: Feb 25, 6pm-12am)

## Summary
- **6 critical bugs fixed** (see BUG_REPORT.md)
- **54/54 local tests passing** (test_dry_run.py)
- **45 SLURM jobs submitted** across 3 clusters
- **Fir cluster down** — 15 jobs redistributed to Narval/Rorqual

## GitHub
- Branch: `v2-full-rewrite`
- Latest commit: `02f8239` — "redistribute 15 Fir jobs to Narval (8) and Rorqual (7)"
- Previous: `2964663` — "fix: resolve 6 critical bugs causing all 45 experiment failures"

## Job Submissions

### Narval (A100-40GB) — 20 jobs total
| Job ID | Backbone | Split | Seed | Status |
|--------|----------|-------|------|--------|
| 57045459 | Caduceus | A | 42 | PENDING |
| 57045460 | Caduceus | A | 123 | PENDING |
| 57045461 | Caduceus | A | 456 | PENDING |
| 57045462 | Caduceus | B | 42 | PENDING |
| 57045463 | Caduceus | B | 123 | PENDING |
| 57045464 | Caduceus | B | 456 | PENDING |
| 57045465 | Caduceus | C | 42 | PENDING |
| 57045466 | Caduceus | C | 123 | PENDING |
| 57045467 | Caduceus | C | 456 | PENDING |
| 57045468 | CNN-GRU | A | 42 | PENDING |
| 57045469 | CNN-GRU | B | 42 | PENDING |
| 57045470 | CNN-GRU | C | 42 | PENDING |
| 57045473 | Evo | A | 42 | PENDING (from Fir) |
| 57045474 | Evo | A | 123 | PENDING (from Fir) |
| 57045475 | Evo | A | 456 | PENDING (from Fir) |
| 57045476 | Evo | B | 42 | PENDING (from Fir) |
| 57045477 | Evo | C | 42 | PENDING (from Fir) |
| 57045478 | NT | A | 42 | PENDING (from Fir) |
| 57045479 | NT | A | 456 | PENDING (from Fir) |
| 57045480 | NT | C | 42 | PENDING (from Fir) |

### Rorqual (H100-80GB) — 19 jobs total
| Job ID | Backbone | Split | Seed | Status |
|--------|----------|-------|------|--------|
| 7372201 | DNABERT-2 | A | 42 | RUNNING |
| 7372202 | DNABERT-2 | A | 123 | PENDING |
| 7372203 | DNABERT-2 | A | 456 | RUNNING |
| 7372204 | DNABERT-2 | B | 42 | RUNNING |
| 7372205 | DNABERT-2 | B | 123 | PENDING |
| 7372206 | DNABERT-2 | B | 456 | RUNNING |
| 7372207 | DNABERT-2 | C | 42 | RUNNING |
| 7372208 | DNABERT-2 | C | 123 | PENDING |
| 7372209 | DNABERT-2 | C | 456 | RUNNING |
| 7372210 | NT | A | 123 | RUNNING |
| 7372211 | NT | B | 123 | RUNNING |
| 7372212 | NT | C | 123 | RUNNING |
| 7372235 | Evo | B | 123 | PENDING (from Fir) |
| 7372236 | Evo | B | 456 | PENDING (from Fir) |
| 7372237 | Evo | C | 123 | PENDING (from Fir) |
| 7372238 | Evo | C | 456 | PENDING (from Fir) |
| 7372239 | NT | B | 42 | PENDING (from Fir) |
| 7372240 | NT | B | 456 | PENDING (from Fir) |
| 7372241 | NT | C | 456 | PENDING (from Fir) |

### Nibi (GPU) — 6 jobs total
| Job ID | Backbone | Split | Seed | Status |
|--------|----------|-------|------|--------|
| 9334978 | CNN-GRU | A | 123 | PENDING |
| 9334979 | CNN-GRU | A | 456 | PENDING |
| 9334980 | CNN-GRU | B | 123 | PENDING |
| 9334981 | CNN-GRU | B | 456 | PENDING |
| 9334982 | CNN-GRU | C | 123 | PENDING |
| 9334983 | CNN-GRU | C | 456 | PENDING |

## Cluster Status
| Cluster | Status | Jobs | Notes |
|---------|--------|------|-------|
| Narval | ACTIVE | 20 | A100-40GB, PENDING queue |
| Rorqual | ACTIVE | 19 | H100-80GB, 9 RUNNING |
| Nibi | ACTIVE | 6 | PENDING queue |
| Fir | DOWN | 0 | Connection refused, all jobs redistributed |
| Killarney | INACTIVE | 0 | No SLURM account association |
| Béluga | INACTIVE | 0 | SLURM plugin incompatibility |

## Experiment Coverage (45/45)
| Backbone | Split A (42,123,456) | Split B (42,123,456) | Split C (42,123,456) |
|----------|---------------------|---------------------|---------------------|
| CNN-GRU | narval/nibi/nibi | narval/nibi/nibi | narval/nibi/nibi |
| Caduceus | narval/narval/narval | narval/narval/narval | narval/narval/narval |
| DNABERT-2 | rorqual/rorqual/rorqual | rorqual/rorqual/rorqual | rorqual/rorqual/rorqual |
| NT | narval/rorqual/narval | rorqual/rorqual/rorqual | narval/rorqual/rorqual |
| Evo | narval/narval/narval | narval/rorqual/rorqual | narval/rorqual/rorqual |

## Expected Timeline
- CNN-GRU: ~2-4 hours (smallest model, ~2M params)
- Caduceus: ~4-6 hours (~7M params)
- Evo: ~8-12 hours (~14M params)
- DNABERT-2: ~8-12 hours (~117M params)
- Nucleotide Transformer: ~12-18 hours (~500M params)
