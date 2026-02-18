"""Prototype causal head modules for causal-effect-aware outputs.

These are research prototypes and intentionally minimal. They provide a
lightweight interface that can be expanded with do-calculus / SCM components.
"""
import torch
import torch.nn as nn


class CausalHead(nn.Module):
    """Simple causal projection head.

    Accepts embedding vectors and outputs a scalar prediction while allowing
    optional conditioning on observed confounders.
    """
    def __init__(self, in_dim: int, confounder_dim: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.confounder_dim = confounder_dim
        total = in_dim + confounder_dim
        self.net = nn.Sequential(nn.Linear(total, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, emb: torch.Tensor, confounders: torch.Tensor = None):
        if confounders is not None:
            x = torch.cat([emb, confounders], dim=-1)
        else:
            x = emb
        return self.net(x).squeeze(-1)
