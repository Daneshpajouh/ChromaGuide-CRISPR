# Implementation Plan — DNABERT-2 + Mamba-2 Primary Path

This document maps the current codebase capabilities, gaps relative to the
targets in `BREAKTHROUGH_RESEARCH_PROMPT.md`, and a prioritized implementation
plan focused on Scenario A (DNABERT-2 + Mamba-2 refinement).

## Current codebase capabilities
- Data loader: `CRISPRoffTDataset` (`src/data/crisprofft.py`)
- Mamba implementations: `src/model/mamba2_block.py`, `src/model/bimamba2_block.py`, `src/model/mamba_deepmens.py`
- DNABERT-2 references and LoRA examples: `emergency_dnabert2_train.py`, `test_mamba_dnabert_compatibility.py`
- Ensemble & RAG infra: `src/train_deepmens_ensemble.py`, `src/model/crispr_rag_tta.py`

## Gaps to reach breakthrough target (ρ > 0.85)
1. End-to-end DNABERT → Adapter → Mamba training pipeline (freeze-foundation, adapter, Mamba, head).
2. Production-ready adapter modules (dimension-checking, LoRA/PEFT support).
3. Tokenization / dataset path producing DNABERT inputs from `CRISPRoffTDataset`.
4. Evaluation harness integrated with the gold test set (Spearman ρ tracking and early stopping).
5. Heterogeneous ensemble fusion (meta-learner) combining DNABERT + Mamba + DeepMEns.
6. Research modules present as placeholders: causal heads, quantum-inspired corrections, topological regularizer (need implementations).
7. Evo-2 integration (Scenario B) — no loader/adapters present.

## Prioritized implementation steps (Scenario A)

Phase 0 — Safety & imports (0.5 day)
- Add `src/__init__.py` and guard fragile imports (MLX) so scripts don't hard-fail.
- Add compatibility alias `train_deepness.py` to avoid CI/typo issues.

Phase 1 — Adapter + DNABERT→Mamba glue (3–5 days)
1. Implement `src/model/adapters.py` with `AdapterFactory`, `LinearAdapter`, and a LoRA placeholder.
2. Implement `src/model/dnabert_mamba.py` (or `src/train_dnabert_mamba.py`) to assemble foundation model + adapter + Mamba stack + head.
3. Add tokenization wrapper so `CRISPRoffTDataset` yields token IDs suitable for DNABERT.

Phase 2 — Evaluation & metrics (1–2 days)
- Integrate `src/data/generate_gold_test_set.py` into training/eval.
- Add Spearman ρ computation and structured JSON logging per epoch.

Phase 3 — HPO & experiments (3–5 days)
- Add Optuna wrapper for adapter size, number of Mamba layers, LR, LoRA hyperparameters.

Phase 4 — Advanced heads (research, 2–4 weeks)
- Implement `src/model/causal_head.py`, `src/model/quantum_correction.py`, `src/model/topo_regularizer.py` as optional heads/regularizers.

Phase 5 — Ensemble fusion (2–4 days)
- Implement `src/model/meta_learner.py` to fuse heterogeneous model outputs and embeddings.

## Files to create / modify (priority order)
- Create: `src/model/adapters.py`
- Create: `src/model/dnabert_mamba.py` or `src/train_dnabert_mamba.py`
- Modify: dataset wrapper to optionally output tokenized inputs
- Create: `src/model/meta_learner.py`
- Optional: `src/model/causal_head.py`, `src/model/quantum_correction.py`, `src/model/topo_regularizer.py`

## Quick wins (actionable now)
1. Add adapter factory + small wrapper — enables DNABERT→Mamba experiments in hours.
2. Add `train_dnabert_mamba.py` using existing model blocks to iterate quickly.
3. Add evaluation + reporting hooks (Spearman) so progress is measurable.

## Deliverables / milestones
- Week 1: Working DNABERT→Mamba training script + adapter module + basic evaluation on mini dataset
- Week 2: HPO experiments, ablation logs, checkpoint saving
- Week 3+: Advanced heads and ensemble fusion prototypes
