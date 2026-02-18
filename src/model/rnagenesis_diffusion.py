import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RNAGenesisDiffusion(nn.Module):
    """
    Latent Diffusion Model.
    Predicts noise epsilon added to latent vector z_t.
    Conditioned on Gene ID (target gene context).
    """
    def __init__(self, latent_dim=256, num_genes=20000, embedding_dim=256):
        super(RNAGenesisDiffusion, self).__init__()

        # Time Embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256)
        )

        # Condition Embedding (Gene ID)
        # In practice, this could be AttO3D GNN embedding, but typically Gene ID for MVP
        self.cond_emb = nn.Embedding(num_genes, embedding_dim)

        # Main Backbone (Residual MLP for 1D data)
        # Input: Latent(256) + Time(256) + Cond(256)
        self.input_proj = nn.Linear(latent_dim, 512)

        self.block1 = ResBlock(512 + 256 + 256, 512)
        self.block2 = ResBlock(512 + 256 + 256, 512)
        self.block3 = ResBlock(512 + 256 + 256, 512)

        self.output_proj = nn.Linear(512, latent_dim)

    def forward(self, x, t, gene_id):
        """
        x: Latent vector z_t (B, 256)
        t: Time step (B, 1) float [0,1] or int embedding?
           Usually t is int [0, 1000], we embed it.
           Here assuming t is normalized or we use sinusoidal.
           Let's use float t and simple MLP embedding for simplicity/SOTA.
        gene_id: (B,) int
        """
        # Time setup
        t_emb = self.time_emb(t) # (B, 256)

        # Cond setup
        c_emb = self.cond_emb(gene_id) # (B, 256)

        h = self.input_proj(x) # (B, 512)

        # Concatenate condition to each block or add?
        # Standard: Add or Concat. We concat.

        # Block 1
        h_cond = torch.cat([h, t_emb, c_emb], dim=1) # 512+256+256
        h = self.block1(h_cond) + h # Residual

        # Block 2
        h_cond = torch.cat([h, t_emb, c_emb], dim=1)
        h = self.block2(h_cond) + h

        # Block 3
        h_cond = torch.cat([h, t_emb, c_emb], dim=1)
        h = self.block3(h_cond) + h

        out = self.output_proj(h)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return h

# Noise Schedule Helper
class DiffusionSchedule:
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.T, (batch_size,), device=device).long()

    def noise_image(self, x_0, t):
        """Adds noise to latent x_0 at timestep t"""
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alphas_cumprod[t]).view(-1, 1)
        epsilon = torch.randn_like(x_0)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon, epsilon
