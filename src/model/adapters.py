"""Adapter modules for connecting foundation models (DNABERT) to Mamba stacks.

Provides a small `AdapterFactory` to create identity/linear adapters and a
LoRA placeholder for future integration with PEFT/LoRA libraries.
"""
from typing import Optional
import torch
import torch.nn as nn


class LinearAdapter(nn.Module):
    """Simple linear projection adapter.

    Projects `in_dim` -> `out_dim` and optionally applies a non-linearity.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: Optional[str] = None):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        if activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = None

    def forward(self, x):
        x = self.proj(x)
        if self.act is not None:
            x = self.act(x)
        return x


class IdentityAdapter(nn.Module):
    def forward(self, x):
        return x


class LoRAAdapter(nn.Module):
    """Lightweight placeholder for LoRA/PEFT-style adapters.

    This intentionally keeps a minimal interface; integrate `peft` or a LoRA
    implementation later to enable parameter-efficient fine-tuning.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int = 4):
        super().__init__()
        # Minimal low-rank parameterization (dense -> low-rank -> dense)
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, out_dim) * 0.01)

    def forward(self, x):
        # x: (..., in_dim)
        # This applies x @ (A @ B)
        ab = torch.matmul(self.A, self.B)
        return x.matmul(ab)


def AdapterFactory(in_dim: int, out_dim: int, kind: str = "linear", **kwargs) -> nn.Module:
    if kind == "linear":
        return LinearAdapter(in_dim, out_dim, activation=kwargs.get("activation"))
    elif kind == "identity":
        return IdentityAdapter()
    elif kind == "lora":
        return LoRAAdapter(in_dim, out_dim, rank=kwargs.get("rank", 4))
    else:
        raise ValueError(f"Unknown adapter kind: {kind}")
