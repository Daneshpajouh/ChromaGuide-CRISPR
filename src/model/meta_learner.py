"""Meta-learner scaffold for fusing heterogeneous model outputs.

The `MetaLearner` accepts a list of model predictions or embeddings and learns
to combine them (simple MLP / attention-based fusion can be added later).
"""
from typing import List
import torch
import torch.nn as nn


class MetaLearner(nn.Module):
    def __init__(self, input_sizes: List[int], hidden: int = 128):
        super().__init__()
        self.input_sizes = input_sizes
        total_dim = sum(input_sizes)
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, inputs: List[torch.Tensor]):
        # Concatenate along last dim
        x = torch.cat(inputs, dim=-1)
        return self.net(x).squeeze(-1)


def build_meta_learner_from_models(models: List[object], hidden: int = 128) -> MetaLearner:
    # Try to infer embedding sizes by querying a .embedding_dim or output shape
    sizes = []
    for m in models:
        s = getattr(m, "embedding_dim", None)
        if s is None:
            # fallback heuristic
            s = getattr(m, "out_dim", None) or getattr(m, "hidden_size", None) or 128
        sizes.append(int(s))
    return MetaLearner(sizes, hidden=hidden)
