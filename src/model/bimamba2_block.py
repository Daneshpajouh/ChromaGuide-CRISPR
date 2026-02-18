from torch import nn
import torch
from src.model.mamba2_block import Mamba2, Mamba2Config

class BiMamba2(nn.Module):
    """
    Bi-Directional Mamba-2 (SSD) Block.
    Inspired by Caduceus (ICML 2024).

    Features:
    - Shared Weights: Uses a single Mamba-2 block for both directions.
    - Efficiency: Linear time complexity (SSD).
    - Fusion: Concatenates forward and backward streams and projects back to d_model.
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config

        # The core Mamba-2 Block (Shared)
        self.mamba = Mamba2(config)

        # Output Projection (2*D -> D) because we concat logic
        # Caduceus often uses 2*D internal, but we project back to D to keep stack size constant.
        # Check if Mamba2 returns same dim? Yes (B, L, D).
        self.out_proj = nn.Linear(config.d_model * 2, config.d_model)

    def forward(self, x, h=None):
        """
        x: (B, L, D)
        h: Ignored for now (training mode mostly)
        """
        # Forward Pass
        # Mamba2 returns (y, h). We discard h for bi-directional as it's non-causal.
        y_fwd, _ = self.mamba(x)

        # Backward Pass
        # Flip along sequence dimension (dim=1)
        x_rev = torch.flip(x, dims=[1])
        y_rev, _ = self.mamba(x_rev)

        # Flip backward output back to original order
        y_rev = torch.flip(y_rev, dims=[1])

        # Fusion
        # Concatenate along feature dimension
        y_cat = torch.cat([y_fwd, y_rev], dim=-1) # (B, L, 2*D)

        # Project back to d_model
        y = self.out_proj(y_cat)

        return y, None # Interface compatibility
