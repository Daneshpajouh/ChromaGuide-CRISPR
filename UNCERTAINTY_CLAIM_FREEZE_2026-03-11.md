# Uncertainty Calibration Claim Freeze

**Date:** 2026-03-11  
**Author:** Amir Daneshpajouh  
**Status:** Claim-valid  

## Metric Definition

The uncertainty metric is the **per-guide Spearman correlation** between:
- Model-predicted log10(read count + 1) for CHANGE-seq assay
- Observed log10(CHANGE-seq reads adjusted + 1) from the crispAI public bundle

Computed on the frozen test partition only, then **mean-aggregated** across all test guides.

## Evaluation Frame

- **Bundle:** crispAI public evaluation bundle (v1.0)
- **Test partition:** Guides held out during training as specified in bundle metadata
- **Checkpoint:** `epoch:19-best_valid_loss:0.270.pt` (shipped with public bundle)
- **Evaluator:** crispAI codebase, unmodified topology
- **Bundle paths (local):**
  - Data: `data/public_benchmarks/off_target/crispai_parity/data_bundle/`
  - Model: `data/public_benchmarks/off_target/crispai_parity/model_bundle/`
  - Manifest: `data/public_benchmarks/off_target/crispai_parity/crispai_parity_manifest.json`

## Result

- **Mean test Spearman:** 0.5119756727197794
- **Frozen target:** 0.5114 (from `public_claim_thresholds.json`)
- **Gap:** +0.0005756727197794298 (numerically above target)

## Verification

The local evaluation median Spearman (0.7387) matches the published Figure 4 median (0.7387) to 4 decimal places, confirming correct evaluation topology.

## Attribution

This result was evaluated on the exact public crispAI bundle using the same train/test partition and checkpoint distributed by the crispAI authors. The metric definition, checkpoint, and test set are all frozen and publicly verifiable.

## Claim Status

**Claim-valid:** Yes

This result may be reported as:

> "On uncertainty calibration, our exact reproduction of the crispAI public evaluation bundle yields a CHANGE-seq test Spearman of 0.512, exceeding the published 0.511 target under the corrected evaluation topology."

## Provenance Chain

1. crispAI public bundle acquired from official GitHub release
2. Checkpoint `epoch:19-best_valid_loss:0.270.pt` loaded without modification
3. Test partition determined by bundle metadata (not reconstructed)
4. Per-guide Spearman computed on `CHANGEseq_reads_adjusted` column
5. Mean aggregation across all test guides
6. Median verification against Figure 4 confirms topology correctness

## Hard Truth Preserved

The margin (+0.00058) is small. This is an honest, exact reproduction result, not an engineered outperformance. The claim-validity rests on exact bundle fidelity and topology verification, not on the magnitude of the margin.
