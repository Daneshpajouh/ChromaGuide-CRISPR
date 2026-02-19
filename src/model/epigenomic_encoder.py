import torch
import torch.nn as nn
import torch.nn.functional as F

class EpigenomicMLPEncoder(nn.Module):
    """
    Dissertation Chapter 4: Epigenomic Feature Encoder.

    Transforms raw epigenomic signals (DNase, ChIP-seq, etc.) from multiple
    modalities into a dense latent representation compatible with the
    DNA backbone's embedding space.

    Architecture:
    - 3-layer MLP with increasing then decreasing width.
    - LayerNorm for signal stabilization.
    - Dropout for regularization as per HPO results (Phase 1).
    """
    def __init__(self, input_dim=9, hidden_dims=[64, 128, 64], d_model=128, dropout=0.3):
        super().__init__()

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        # Final projection to d_model (or whatever the main embedding space is)
        self.encoder = nn.Sequential(*layers)
        self.out_proj = nn.Linear(current_dim, d_model)

    def forward(self, x):
        """
        x: [Batch, SeqLen, InputDim]
           InputDim is the number of epigenomic tracks (e.g., 9)
        """
        # We handle the input per-position across the sequence
        # x shape: [B, L, I]
        batch, seq_len, _ = x.shape

        # Flatten time/position dimension to use MLP
        x_flat = x.view(-1, x.shape[-1]) # [B*L, I]

        latent_flat = self.encoder(x_flat) # [B*L, 64]
        out_flat = self.out_proj(latent_flat) # [B*L, d_model]

        # Reshape back to sequence
        return out_flat.view(batch, seq_len, -1)

class GatedMultimodalFusion(nn.Module):
    """
    Dissertation Chapter 4: Gated Multimodal Fusion.

    oselectively integrates epigenomic features from the MLP encoder
    with sequence features from the DNABERT/Mamba backbone.

    Mechanism:
    z_fused = (1 - g) * z_seq + g * z_epi
    where g = sigmoid(W[z_seq; z_epi])
    """
    def __init__(self, d_model):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.fusion_mix = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_seq, z_epi):
        """
        z_seq: [B, L, D]
        z_epi: [B, L, D] (output from EpigenomicMLPEncoder)
        """
        combined = torch.cat([z_seq, z_epi], dim=-1) # [B, L, 2D]

        # Compute dynamic gating weights
        gate = self.gate_proj(combined) # [B, L, D]

        # Blend features
        fused = (1 - gate) * z_seq + gate * z_epi

        return self.norm(fused)
