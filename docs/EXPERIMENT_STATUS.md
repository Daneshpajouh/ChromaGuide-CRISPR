# ChromaGuide v2 — Experiment Deployment Status

> **Last Updated:** February 26, 2026, 7:15 AM UTC (Feb 25, 11:15 PM PST)  
> **Branch:** `v2-full-rewrite`  
> **Phase:** Synthetic Data Baseline (Phase 1 of 3)  
> **Status:** 🔴 ALL 45 JOBS FAILED — bugs found and fixed, awaiting resubmission

---

## Executive Summary

**All 45 SLURM jobs have FAILED** across all 4 active clusters due to critical bugs in the training code. A comprehensive audit uncovered **6 bugs** (3 critical), all of which have been fixed and verified locally with a 54-test dry-run suite passing on all 5 backbones.

### Bug Summary

| # | Severity | File | Issue | Status |
|---|----------|------|-------|--------|
| 1 | CRITICAL | `train_experiment.py:340` | `total_mem` → `total_memory` (AttributeError) | ✅ FIXED |
| 2 | CRITICAL | 15 redistributed SLURM scripts | Missing `--backbone` argument | ✅ FIXED |
| 3 | MEDIUM | `reproducibility.py:25` | Same `total_mem` → `total_memory` bug | ✅ FIXED |
| 4 | CRITICAL | `train_experiment.py` | Raw DNA sequences not passed to transformer encoders | ✅ FIXED |
| 5 | CRITICAL | `chromaguide.py:70` | `nucleotide_transformer` missing from `_needs_raw_sequences` | ✅ FIXED |
| 6 | MINOR | `modules/__init__.py` | Missing `NucleotideTransformerEncoder` export | ✅ FIXED |

### Verification

- **54 local tests passed** across all 5 backbones (imports, forward pass, loss, backward, conformal, scheduler)
- All 45 SLURM scripts verified for `--backbone` argument, correct paths, and partition settings
- Fir scripts confirmed with `gpubase_bygpu_b3` partition

---

## Next Steps

1. Push all fixes to GitHub (`v2-full-rewrite` branch)
2. Deploy updated code to all 4 working clusters via `scp`
3. Resubmit all 45 jobs
4. Monitor completion (~4–18 hours depending on backbone)

---

## Cluster Status

### Active Clusters

| Cluster | GPU | Jobs | Previous Status | Next Action |
|---------|-----|------|-----------------|-------------|
| **Narval** | A100-40GB | 12 | ALL FAILED (`total_mem` + missing `--backbone`) | Redeploy + resubmit |
| **Rorqual** | H100-80GB | 12 | ALL LIKELY FAILED (same bugs) | Redeploy + resubmit |
| **Fir** | A100-80GB | 15 | ALL LIKELY FAILED (same bugs) | Redeploy + resubmit |
| **Nibi** | GPU | 6 | ALL LIKELY FAILED (same bugs) | Redeploy + resubmit |

### Inactive Clusters

| Cluster | Issue | Resolution |
|---------|-------|------------|
| **Killarney** | No SLURM account association for `amird` | 9 jobs redistributed |
| **Béluga** | SLURM plugin incompatibility | 6 jobs redistributed |

---

## Job Distribution by Backbone

### CNN-GRU (~2M parameters)
| Split | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| A | Narval | Nibi | Nibi |
| B | Narval | Nibi | Nibi |
| C | Narval | Nibi | Nibi |

### Caduceus (~7M parameters)
| Split | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| A | Narval | Narval | Narval |
| B | Narval | Narval | Narval |
| C | Narval | Narval | Narval |

### DNABERT-2 (~117M parameters)
| Split | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| A | Rorqual | Rorqual | Rorqual |
| B | Rorqual | Rorqual | Rorqual |
| C | Rorqual | Rorqual | Rorqual |

### Nucleotide Transformer (~500M parameters)
| Split | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| A | Fir | Rorqual | Fir |
| B | Fir | Rorqual | Fir |
| C | Fir | Rorqual | Fir |

### Evo (~14M parameters)
| Split | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| A | Fir | Fir | Fir |
| B | Fir | Fir | Fir |
| C | Fir | Fir | Fir |

---

## Performance Targets (from PhD Proposal)

| Metric | Target | SOTA Reference |
|--------|--------|----------------|
| Spearman ρ (gene-held-out) | ≥ 0.91 | PLM-CRISPR: 0.950 |
| Off-target AUROC | ≥ 0.92 | — |
| Conformal coverage | 90% ± 2% | — |
| ECE | < 0.05 | — |
| Statistical significance | p < 0.001 | — |

---

## Monitoring Commands

```bash
# Check job status on each cluster
ssh narval "squeue -u amird"
ssh rorqual "squeue -u amird"
ssh fir "export PATH=/opt/software/slurm-24.11.6/bin:$PATH && squeue -u amird"
ssh nibi "squeue -u amird"

# Check completed results
ssh <cluster> "ls ~/scratch/chromaguide_v2/results/*/results.json 2>/dev/null | wc -l"
```
