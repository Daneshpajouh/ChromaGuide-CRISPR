#!/bin/bash
# Test loss function specifically

cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2
source ~/projects/def-kwiese/amird/pytorch_env/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "=== Testing Loss Function ==="
python3 << 'EOF'
import torch
from src.data.crisprofft import CRISPRoffTDataset
from src.model.crispro import CRISPROModel
from src.utils.loss import CombinedLoss
from torch.utils.data import DataLoader, Subset

device = "cuda"
print(f"Device: {device}")

# Load data
dataset = CRISPRoffTDataset(use_mini=True, context_window=4096)
subset = Subset(dataset, range(4))
loader = DataLoader(subset, batch_size=2)
batch = next(iter(loader))

seqs = batch['sequence'].to(device)
epi = batch['epigenetics'].to(device)
targets_raw = batch['efficiency'].to(device, dtype=torch.float32)

print(f"\nTargets: {targets_raw}")

# Create targets
targets_cls = (targets_raw > 0).float()
targets_reg = targets_raw / 100.0
mask_active = (targets_cls > 0).float()

print(f"targets_cls: {targets_cls}")
print(f"targets_reg: {targets_reg}")
print(f"mask_active: {mask_active}, sum={mask_active.sum()}")

# Initialize model
model = CRISPROModel(d_model=256, n_layers=4, n_modalities=6).to(device)
criterion = CombinedLoss()

# Forward pass
with torch.no_grad():
    pred_cls, pred_reg = model(seqs, epi)

print(f"\nPredictions:")
print(f"  pred_cls: {pred_cls.flatten()}")
print(f"  pred_reg: {pred_reg.flatten()}")
print(f"  pred_cls NaN: {torch.isnan(pred_cls).any()}")
print(f"  pred_reg NaN: {torch.isnan(pred_reg).any()}")

# Test loss function
print(f"\n=== Testing Loss Components ===")
try:
    # The actual loss calculation from train.py
    from src.utils.loss import FocalLoss, DiceLoss
    import torch.nn as nn

    criterion_cls = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_reg = nn.SmoothL1Loss()

    # Classification loss
    loss_cls = criterion_cls(pred_cls, targets_cls)
    print(f"Classification loss: {loss_cls.item()}")
    print(f"  NaN: {torch.isnan(loss_cls).any()}")

    # Regression loss (masked)
    if mask_active.sum() > 0:
        loss_reg = criterion_reg(pred_reg[mask_active.bool()], targets_reg[mask_active.bool()])
        print(f"Regression loss: {loss_reg.item()}")
        print(f"  NaN: {torch.isnan(loss_reg).any()}")
    else:
        loss_reg = torch.tensor(0.0, device=device)
        print(f"Regression loss: 0.0 (no active samples)")

    # Combined
    loss = loss_cls + 0.5 * loss_reg
    print(f"\nCombined loss: {loss.item()}")
    print(f"  NaN: {torch.isnan(loss).any()}")

    # Try backward
    print(f"\n=== Testing Backward Pass ===")
    loss.backward()
    print("Backward pass completed successfully!")

    # Check gradients
    nan_grads = 0
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            nan_grads += 1
            print(f"  ❌ NaN gradient in: {name}")

    if nan_grads == 0:
        print("  ✅ All gradients OK!")
    else:
        print(f"  ❌ {nan_grads} parameters have NaN gradients")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

EOF
