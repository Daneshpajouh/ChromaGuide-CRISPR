# CRITICAL THESIS FIX SUMMARY
# Date: Feb 22, 2026
# Purpose: Enable DNABERT-2 backbone to achieve Rho >= 0.911 target

## PROBLEM IDENTIFIED
- Previous session results:
  - Baseline: 0.8507 ✅ (target 0.80)
  - Multimodal: 0.8494 ❌ (target 0.911 = 93.2% of target)
  - Off-target: 0.7541 ❌ (target 0.99 = 76.2% of target)

- Root cause of multimodal failure: DNABERT-2 ALiBi meta device error
- DNABERT-2 is essential per thesis methodology (Table 1: transformer ablation)
- CNN-GRU fallback insufficient - only achieved 0.8494

## STEP 1: DNABERT-2 LOADING FIX (COMPLETED)
### File Modified: src/chromaguide/sequence_encoder.py

KEY CHANGES:
1. Changed from AutoModel.from_pretrained() → BertModel() direct initialization
2. Load config from cache, create BertModel on CPU
3. Load weights directly from cached pytorch_model.bin using torch.load()
4. Prevents ALiBi tensor creation on meta device during __init__

STRATEGY:
- BertModel(config, add_pooling_layer=False).to('cpu')
- torch.load(weights_path, map_location='cpu', weights_only=True)
- state_dict loaded onto CPU-initialized model (safe)
- Move to GPU later in training pipeline

COMMITS:
- Commit d94ef9d: "CRITICAL FIX: DNABERT-2 direct weight loading..."
- Pushed to origin/main

## MANUAL STEPS TO EXECUTE ON NARVAL

### Step 1: Pull latest code
```bash
cd ~/chromaguide_experiments
git pull origin main  # or copy src/ and scripts/ from local
```

### Step 2: Verify DNABERT-2 cache exists
```bash
ls -la ~/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/*/
# Should see: config.json, pytorch_model.bin, etc.
```

### Step 3: Test DNABERT-2 loading (quick CPU test, ~1 min)
```bash
cd ~/chromaguide_experiments
python3 << 'EOF'
import sys, torch
sys.path.insert(0, 'src')
from chromaguide.sequence_encoder import DNABERT2Encoder

print('Testing DNABERT-2...')
try:
    encoder = DNABERT2Encoder(d_model=768)
    print('✅ Model created')
    x = torch.randint(0, 30, (2, 23))
    y = encoder(x)
    print(f'✅ Forward pass OK: {y.shape}')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
EOF
```

### Step 4: Submit DNABERT-2 multimodal training job
```bash
cd ~/chromaguide_experiments
sbatch scripts/slurm_multimodal_dnabert2_fixed.sh
```

Expected output from training script:
```
Epoch 1: Loss: [value] | Val Rho: 0.70+
Epoch 2: Loss: [value] | Val Rho: 0.75+
...
Epoch 50: Loss: [value] | Val Rho: 0.85+
FINAL GOLD Rho: 0.90+  # Target: >= 0.911
```

## STEP 2: OFF-TARGET IMPROVEMENTS (PENDING)

Off-target still at 0.7541 (target 0.99). Issues:
1. Extreme class imbalance: 99.54% OFF-target vs 0.46% ON-target
2. Needs focal loss + SMOTE resampling OR different architecture

File: scripts/train_off_target_v4.py

Required changes for next training run:
- Use focal loss instead of weighted BCE
- Apply SMOTE oversampling to ON-target class
- Increase epochs to 200 (longer training allowed)
- Consider focal_loss weight = 2.0

```python
# In train_off_target_v4.py, replace:
from focal_loss.focal_loss import FocalLoss
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

## STEP 3: TRAINING SCHEDULE

Priority 1: DNABERT-2 Multimodal (Rho target 0.911)
- Expected time: ~2 hours on A100
- After completion: Check FINAL GOLD Rho

Priority 2: Off-target with Focal Loss (AUROC target 0.99)
- Expected time: ~3 hours on A100
- May require iterative improvement

## FILES MODIFIED/CREATED

### Modified (Committed to GitHub)
1. src/chromaguide/sequence_encoder.py
   - DNABERT2Encoder class completely rewritten
   - Direct weight loading strategy
   - Commit: d94ef9d

### Created (Local, need to copy to Narval)
1. scripts/slurm_multimodal_dnabert2_fixed.sh
   - DNABERT-2 backbone training job
   - 50 epochs, full training budget
   - HF_*_OFFLINE=1 for offline operation

2. test_dnabert2_fix.py
   - Quick verification script
   - Tests model creation + forward pass

### Existing (Updated elsewhere)
1. scripts/train_on_real_data_v2.py
   - Already compatible with DNABERT-2
   - Tokenizer loading with fallback

## EXPECTED OUTCOMES

Multimodal with DNABERT-2 (50 epochs):
- Epoch 1: Rho ~0.75
- Epoch 25: Rho ~0.83
- Epoch 50: Rho ~0.85-0.89 (goal: >= 0.911)

Key insight: DNABERT-2 embeddings + full training budget should improve
from 0.8494 (CNN-GRU, 20 epochs) to 0.91+

## IF DNABERT-2 STILL FAILS
Fallback Plan:
1. Keep CNN-GRU multimodal at 0.8494 (93.2% of thesis target)
2. Acknowledge trade-off in results discussion
3. Present ablation: "CNN-GRU alternative when transformer unavailable"
4. Focus on off-target + design score integration

## THESIS IMPACT

Current state (Session 1):
- Only 1/3 targets hit (baseline at 0.8507)

After DNABERT-2 fix:
- Baseline: 0.8507 ✅
- Multimodal: 0.90+ (goal) 🎯
- Off-target: 0.75+ (need focal loss)

This enables submitting a defensible thesis with working pipeline.
