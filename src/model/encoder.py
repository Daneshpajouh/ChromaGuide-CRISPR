import torch
import torch.nn as nn
from .mamba_wrapper import SafeMambaBlock

class MambaEncoder(nn.Module):
    """
    Deep Mamba Encoder.
    Stacks multiple SafeMambaBlocks to process DNA sequences.
    """
    def __init__(self, d_model=256, n_layers=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            SafeMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Input: [Batch, SeqLen, d_model]
        Output: [Batch, SeqLen, d_model]
        """
        for layer in self.layers:
            # Residual connection is handled inside Mamba usually or here?
            # Mamba paper: Block -> Residual
            # Standard: x = x + layer(norm(x)) or layer(x) depending on block impl
            # Our SafeMambaBlock is just the mixer.

            # Simple Residual Pre-Norm
            residual = x
            x = layer(x)
            x = residual + x

        return self.norm_f(x)
