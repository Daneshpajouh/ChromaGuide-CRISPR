#!/bin/bash
# Comprehensive NaN debugging script
# Run diagnostic checks on model initialization, forward pass, and loss

cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2
source ~/projects/def-kwiese/amird/pytorch_env/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "=== Comprehensive NaN Debugging ==="
python3 << 'EOF'
import torch
from src.data.crisprofft import CRISPRoffTDataset
from src.model.crispro import CRISPROModel
from src.utils.loss import CombinedLoss
import numpy as np

device = "cuda"
print(f"Device: {device}")

# 1. Load dataset (micro mode)
print("\n=== Loading Data ===")
dataset = CRISPRoffTDataset(use_mini=True, context_window=4096)
from torch.utils.data import Subset
subset = Subset(dataset, range(4))  # Just 4 samples
from torch.utils.data import DataLoader
loader = DataLoader(subset, batch_size=2, shuffle=False)
batch = next(iter(loader))
print(f"Batch loaded: seqs shape={batch['sequence'].shape}")

# 2. Initialize model
print("\n=== Initializing Model ===")
model = CRISPROModel(d_model=256, n_layers=4, n_modalities=6).to(device)

# Check model weights
nan_params = sum([torch.isnan(p).any().item() for p in model.parameters()])
inf_params = sum([torch.isinf(p).any().item() for p in model.parameters()])
print(f"NaN in model parameters: {nan_params}")
print(f"Inf in model parameters: {inf_params}")

# 3. Forward pass
print("\n=== Forward Pass ===")
seqs = batch['sequence'].to(device)
epi = batch['epigenetics'].to(device)
targets = batch['efficiency'].to(device, dtype=torch.float32)

print(f"Input seqs - NaN: {torch.isnan(seqs).any()}, Inf: {torch.isinf(seqs).any()}")
print(f"Input epi - NaN: {torch.isnan(epi).any()}, Inf: {torch.isinf(epi).any()}")
print(f"Targets - NaN: {torch.isnan(targets).any()}, Inf: {torch.isinf(targets).any()}")
print(f"Targets values: {targets}")

with torch.no_grad():
    pred_cls, pred_reg = model(seqs, epi)
    print(f"Predictions cls - NaN: {torch.isnan(pred_cls).any()}, Inf: {torch.isinf(pred_cls).any()}")
    print(f"Predictions reg - NaN: {torch.isnan(pred_reg).any()}, Inf: {torch.isinf(pred_reg).any()}")
    print(f"Pred cls range: [{pred_cls.min():.4f}, {pred_cls.max():.4f}]")
    print(f"Pred reg range: [{pred_reg.min():.4f}, {pred_reg.max():.4f}]")

# 4. Loss calculation
print("\n=== Loss Calculation ===")
criterion = CombinedLoss()
targets_cls = (targets > 0).float()
targets_reg = targets / 100.0
mask_active = (targets_cls > 0).float()

print(f"targets_cls: {targets_cls}")
print(f"targets_reg: {targets_reg}")
print(f"mask_active: {mask_active}, sum={mask_active.sum()}")

try:
    loss = criterion(pred_cls, pred_reg, targets_cls, targets_reg)
    print(f"Loss: {loss.item()}")
    print(f"Loss NaN: {torch.isnan(loss).any()}")
except Exception as e:
    print(f"Loss calculation error: {e}")

EOF
