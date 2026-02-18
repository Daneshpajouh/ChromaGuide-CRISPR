import torch
import torch.nn as nn

class EpigenomicFusion(nn.Module):
    """
    Epigenomic Fusion Layer.
    Injects 1D epigenomic tracks (ATAC, Methylation, CTCF) into the DNA latent state.

    Mechanism:
    Project(Concat(DNA_State, Epigenome)) -> New_State
    """
    def __init__(self, d_model=256, n_modalities=5):
        super().__init__()
        self.n_modalities = n_modalities

        # We assume epigenomic data is passed as a dense vector per position
        # For simplicity in Phase 2, we project the modalities to distinct features
        self.epi_proj = nn.Linear(n_modalities, d_model)

        # Fusion Gate: Allows the model to selectively ignore epigenetics if irrelevant
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # Output projection
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dna_state, epi_tracks):
        """
        dna_state: [Batch, SeqLen, d_model]
        epi_tracks: [Batch, SeqLen, n_modalities]
        """
        # 1. Project Epigenomic modalities to model dimension
        epi_emb = self.epi_proj(epi_tracks) # [B, L, D]

        # 2. Concatenate
        combined = torch.cat([dna_state, epi_emb], dim=-1) # [B, L, 2D]

        # 3. Gating Mechanism
        gate = self.fusion_gate(combined)

        # 4. Fusion
        # Fusing: (DNA + Gate * Epi) or similar.
        # Here we do a learned linear mixing weighted by the gate
        fused = self.out_proj(combined) * gate

        return self.norm(fused + dna_state) # Residual connection
