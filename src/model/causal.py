import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDecoder(nn.Module):
    """
    Day 2: The "Causality" Engine.
    Structural Causal Model (SCM) Head for CRISPRO.
    Disentangles Sequnce, Chromatin, and Accessibility.
    Supports Counterfactual Intervention.
    """
    def __init__(self, d_model=256):
        super().__init__()

        # Sub-encoders for disentanglement
        self.seq_encoder = nn.Linear(d_model, 64)   # H_Sequence
        self.chrom_encoder = nn.Linear(d_model, 64) # H_Chromatin
        self.access_encoder = nn.Linear(d_model, 32) # H_Accessibility

        # Structural Equations (MLPs)
        # 1. Sequence -> PAM Affinity (P)
        self.f_PAM = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )

        # 2. Chromatin + CellType -> Accessibility (A)
        # Note: We simulate CellType as latent noise or feature for now
        self.f_Access = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )

        # 3. PAM (P) + Accessibility (A) -> R-Loop (R)
        self.f_Rloop = nn.Sequential(
            nn.Linear(32+32, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )

        # 4. R-Loop (R) -> Cleavage (Y)
        self.f_Cleave = nn.Linear(32, 1)

    def forward(self, h, intervention=None):
        """
        h: [B, d_model] Latent representation from Mamba Backbone
        intervention: dict {'A': tensor_val} for counterfactuals
        """
        # Disentangle
        h_S = self.seq_encoder(h)
        h_C = self.chrom_encoder(h)

        # Causal Flow
        # P = f(S)
        h_P = self.f_PAM(h_S)

        # A = f(C)
        h_A = self.f_Access(h_C)

        # Intervention logic
        if intervention is not None and 'A' in intervention:
            # Overwrite A with counterfactual value
            # Assume intervention['A'] is [B, 32] or scalar broadcast
            h_A = intervention['A']

        # R = f(P, A)
        h_R = self.f_Rloop(torch.cat([h_P, h_A], dim=-1))

        # Y = f(R)
        logits = self.f_Cleave(h_R)

        return logits, (h_S, h_C, h_P, h_A, h_R)
