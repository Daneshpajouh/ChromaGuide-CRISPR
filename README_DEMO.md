# 🧪 CRISPRO-MAMBA-X: Advanced Features Guide

## 1. Zero-Shot Benchmarking
Evaluate the model's ability to generalize from Human to Mouse data.
```bash
python3 src/analysis/run_benchmark.py --checkpoint checkpoints/best_model.pth
```
*Note*: Requires full dataset or correctly formatted mini-dataset with 'Species' column.

## 2. Causal Introspection (Counterfactuals)
Run "What If" scenarios to measure the causal effect of Chromatin Accessibility on Efficiency.
```bash
python3 src/analysis/run_counterfactual.py --checkpoint checkpoints/best_model.pth --data data/mini/crisprofft/mini_crisprofft.txt
# Output: ATE (Average Treatment Effect) for Open vs Closed Chromatin
```

## 3. Few-Shot Fine-Tuning (Novelty)
Adapt the trained Foundation Model to a new Cas12a dataset (simulated) by freezing the backbone and training the head.
```bash
python3 src/finetune.py --checkpoint checkpoints/best_model.pth --data data/mini/cas12a/mini_cas12a.txt --epochs 3
# Output: checkpoints/adapters/cas12a_adapter.pth
```

## 4. Mechanistic Audit
Verify the internal representations align with Biophysics (Information Theory/Thermo).
```bash
python3 src/analysis/run_mechanistic_audit.py --checkpoint checkpoints/best_model.pth
```
