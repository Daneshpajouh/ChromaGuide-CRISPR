import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermodynamicLoss(nn.Module):
    """
    Chapter 5: Thermodynamic Information Geometry.

    Implements the Thermodynamic Variational Objective (TVO) or Free Energy Loss.
    Hypothesis: The training process mimics the minimization of Helmholtz Free Energy (F).

    F = U - TS
    Loss = Energy (Reconstruction/Prediction Error) - Temperature * Entropy (Regularization)

    Equation: L = MSE + beta * (-H(z))
    Or Variational: L = -ELBO = - (E[log p(x|z)] - KL(q(z|x) || p(z)))
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.T = temperature

    def forward(self, logits, targets, latents):
        """
        logits: (B, 1) or (B, C) - The 'Energy' components
        targets: (B, 1) or (B, C)
        latents: (B, D) - The internal state
        """
        # 1. Internal Energy (U): Prediction Error
        # Corresponds to likelihood term
        if targets.dim() == logits.dim():
             # Regression / Binary
             energy_loss = F.mse_loss(logits, targets) if logits.size(1) == 1 else F.cross_entropy(logits, targets)
        else:
             energy_loss = F.mse_loss(logits, targets.float())

        # 2. Entropy (S): Information Content of Latents
        # We approximate Entropy H(z) via differential entropy estimate or simply L2 norm (Gaussian prior assumption)
        # KL(q || N(0,1)) approx sum(z^2)
        entropy_term = torch.mean(latents ** 2)

        # 3. Free Energy (F = U + TS) note sign change for minimization
        # We want to Minimize F.
        # F = U - TS?
        # Actually in Information Bottleneck: L = I(X;Z) - beta * I(Z;Y) ...
        # In VAE: L = Recon + Beta * KL.
        # Here: U is Recon. KL is Complexity cost (Energy to encode).
        # So L = U + T * KL.

        free_energy = energy_loss + self.T * entropy_term

        return free_energy, energy_loss, entropy_term
