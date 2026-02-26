"""Reproducibility utilities."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_M": f"{total / 1e6:.2f}M",
        "trainable_M": f"{trainable / 1e6:.2f}M",
    }
