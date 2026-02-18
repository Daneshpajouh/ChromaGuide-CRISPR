# Implementation Plan — DNABERT-2 + Mamba-2 Integration (HybriDNA-style)

Goal
----
Deliver a production-ready DNABERT-2 -> Adapter -> Mamba-2 training and evaluation pipeline supporting multi-task heads (on-target efficacy, off-target risk, indel prediction) with parameter-efficient fine-tuning options and device-aware execution (CUDA / MPS / CPU).

Phases & Tasks
----------------
1. Scaffolding & Safe Imports (done)
   - Add guarded DNABERT encoder wrapper and Mamba scaffold.
   - Implement Adapter + LoRA placeholder.
   - Add DeviceManager for automatic device selection and tensor/model placement.

2. Training Script (this deliverable)
   - `src/train_dnabert_mamba.py`: tokenization, adapter, mamba, head, training loop, checkpointing, early stopping, metrics.
   - Synthetic dataset generator for smoke tests.

3. Data Integration
   - Parsers for GUIDE-seq, CIRCLE-seq, DeepHF datasets.
   - Data augmentations and PAM-aware off-target candidate enumerator.

4. Fine-tuning & Parameter Efficiency
   - Integrate LoRA adapters into DNABERT when fine-tuning is required.
   - Add adapter fusion and gradual unfreezing schedules.

5. Evaluation & Ablation
   - Use `src/models/chromaguide_unified.py` to run ablation searches over component combinations and fusion strategies.
   - Evaluation targets: Spearman (target ≥ 0.65 for advanced scenarios), AUROC for off-target.

6. Productionization
   - Add robust checkpointing, logging, config management (JSON/YAML), reproducible seeds.
   - Add CI smoke tests (GitHub Actions) that run the synthetic-data smoke training on CPU.

Timeline (estimates)
---------------------
- Scaffolding & training script (current): 1–2 days (completed scaffolding + training script)
- Data parsers + preprocessing: 2–3 days per dataset
- LoRA integration + fine-tuning experiments: 3–5 days
- Ablation & evaluation experiments: 2–4 days
- CI / reproducibility / docs: 1–2 days

Notes
-----
- The current training script supports guarded fallbacks so you can run smoke tests without large transformer downloads. Replace fallbacks with full components when environment and data are ready.
- DeviceManager is used across training & unified models to ensure MPS/CUDA/CPU compatibility.

Next Actions
------------
1. Run a smoke training using the `chromaguide` mamba env and confirm MPS/CUDA selection.
2. Wire real dataset loaders (DeepHF / GUIDE-seq) into training script and add CLI args for dataset selection.
3. Add LoRA integration and run parameter-efficiency experiments.
