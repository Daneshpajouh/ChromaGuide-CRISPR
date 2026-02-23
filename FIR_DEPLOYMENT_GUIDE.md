# FIR CLUSTER DEPLOYMENT - READY TO RUN ✅

## Status: All files transferred to FIR and ready for execution

**Cluster**: FIR (fir.alliancecan.ca)
**Date**: February 22, 2026
**Status**: ✅ READY

---

## Files Deployed to FIR

```
~/chromaguide/
├── src/chromaguide/
│   └── sequence_encoder.py         ✅ DNABERT-2 fix
├── scripts/
│   ├── train_off_target_focal.py   ✅ Focal loss implementation
│   ├── slurm_fir_multimodal_dnabert2.sh    (SLURM - if available)
│   └── slurm_fir_off_target_focal.sh       (SLURM - if available)
├── test_dnabert2_fix.py            ✅ Quick validation
├── run_training_fir.sh             ✅ Direct execution script
└── data/                           ✅ (already on FIR)
```

---

## How to Run Training on FIR

### Option A: Direct Execution (Recommended for FIR)

If SLURM is not available, use direct execution:

```bash
# 1. SSH to FIR
ssh fir.alliancecan.ca

# 2. Run the training script (handles ~5 hours of compute)
cd ~/chromaguide
bash run_training_fir.sh

# 3. Training will output results to logs/
# Expected:
#   - Multimodal FINAL GOLD Rho: 0.88-0.92 (target 0.911)
#   - Off-target Best AUROC: 0.82-0.88 (target 0.99)
```

**What it does**:
- Sets up Python 3.11 environment with PyTorch
- Creates virtual environment
- Installs dependencies
- Validates DNABERT-2 fix (1 minute)
- Trains multimodal model with DNABERT-2 (2 hours)
- Trains off-target with focal loss (3 hours)

**Total runtime**: ~5 hours

---

### Option B: Background Execution (for longer sessions)

To run in background without keeping SSH session open:

```bash
# SSH to FIR
ssh fir.alliancecan.ca

# Run training in background
cd ~/chromaguide
nohup bash run_training_fir.sh > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Detach from SSH (Ctrl+D)

# Later, reconnect and check progress:
ssh fir.alliancecan.ca
tail -20 logs/training_*.log
tail -20 logs/multimodal_dnabert2_*.log
tail -20 logs/off_target_focal_*.log
```

---

### Option C: SLURM Submission (if available on FIR)

If FIR has SLURM, use:

```bash
cd ~/chromaguide
sbatch scripts/slurm_fir_multimodal_dnabert2.sh
sbatch scripts/slurm_fir_off_target_focal.sh
squeue -u amird
```

---

## Manual Execution (Fallback)

If you prefer to run commands individually:

```bash
ssh fir.alliancecan.ca
cd ~/chromaguide

# Load modules
module purge
module load gcc/12
module load cuda/12.1
module load python/3.11

# Create Python environment
python3 -m venv ~/venvs/chromaguide_manual
source ~/venvs/chromaguide_manual/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate einops biopython scikit-learn pandas numpy

# Set up environment
export PYTHONPATH=~/chromaguide/src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Test DNABERT-2 fix
python test_dnabert2_fix.py

# Train multimodal (50 epochs)
python -u scripts/train_on_real_data_v2.py \
    --backbone dnabert2 \
    --epochs 50 \
    --batch_size 250 \
    --lr 5e-4 \
    --device cuda \
    --seed 42

# Train off-target (200 epochs)
python -u scripts/train_off_target_focal.py \
    --epochs 200 \
    --batch_size 512 \
    --lr 0.0005 \
    --seed 42 \
    --device cuda
```

---

## What Gets Fixed

### 1. **DNABERT-2 Loading** (sequence_encoder.py)
- **Old Problem**: `AutoModel.from_pretrained()` creates ALiBi tensors on meta device
- **New Fix**: Direct `BertModel()` instantiation on CPU + `torch.load()`
- **Expected Improvement**: Multimodal 0.8494 → 0.88-0.92 Rho

### 2. **Off-Target Class Imbalance** (train_off_target_focal.py)
- **Old Problem**: Weighted BCE insufficient for 99.54% class imbalance
- **New Fix**: Focal Loss with γ=2.0, α=0.25
- **Expected Improvement**: Off-target 0.7541 → 0.82-0.88 AUROC

---

## Expected Results

After training completes (~5 hours):

| Model | Current | Expected | Target | Status |
|-------|---------|----------|--------|--------|
| Baseline | 0.8507 | 0.8507 | ≥0.80 | ✅ PASS |
| **Multimodal** | 0.8494 | **0.88-0.92** | **0.911** | 🎯 TARGET |
| **Off-Target** | 0.7541 | **0.82-0.88** | **0.99** | ⚠️ DATA-LIMITED |

---

## Monitoring Progress

### While Training (SSH connected)

```bash
# Watch multimodal training
tail -f logs/multimodal_dnabert2_*.log

# Watch off-target training (in another terminal)
tail -f logs/off_target_focal_*.log
```

### After Training Complete

```bash
# Extract final metrics
grep "FINAL GOLD Rho" logs/multimodal_*.log
grep "Best AUROC" logs/off_target_*.log

# View full loss curves
grep "Epoch" logs/*.log | tail -20
```

---

## Files Summary

### Critical Files Already on FIR
- `~/chromaguide/src/chromaguide/sequence_encoder.py` - DNABERT-2 fix ✅
- `~/chromaguide/scripts/train_off_target_focal.py` - Focal loss ✅
- `~/chromaguide/test_dnabert2_fix.py` - Validation ✅
- `~/chromaguide/run_training_fir.sh` - Direct execution ✅

### On Local Mac (for reference)
- `/Users/studio/Desktop/PhD/Proposal/deploy_package/` - All source files
- `/Users/studio/Desktop/PhD/Proposal/run_training_fir.sh` - Local copy
- `/Users/studio/Desktop/PhD/Proposal/DEPLOYMENT_REPORT_FINAL.md` - Complete documentation

---

## Next Steps

1. **SSH to FIR** and run `bash run_training_fir.sh`
2. **Wait ~5 hours** for both jobs to complete
3. **Extract results** from log files
4. **Compare with targets**:
   - Multimodal: If ≥0.91 Rho → ✅ Success
   - Off-target: If ≥0.85 AUROC → ✅ Good progress

---

## Troubleshooting

### If DNABERT-2 test fails
```bash
# Check HuggingFace cache
ls -la ~/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/

# If missing, manually download (on FIR):
python -c "from transformers import AutoModel; AutoModel.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)"
```

### If training runs out of memory
Lower batch sizes:
```bash
--batch_size 128  # for multimodal
--batch_size 256  # for off-target
```

### If PyTorch GPU/CUDA fails
```bash
# Check GPU availability
nvidia-smi

# Fallback to CPU (slow):
--device cpu
```

---

## Summary

✅ **All fixes deployed to FIR**
✅ **Ready for execution**
⏳ **Awaiting training run** (~5 hours)

**Command to start**: `bash run_training_fir.sh`

Good luck! 🚀
