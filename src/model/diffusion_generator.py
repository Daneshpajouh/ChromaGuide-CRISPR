
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MambaDiff(nn.Module):
    """
    Mamba-based Diffusion Backbone
    Predicts noise epsilon given x_t and t.
    """
    def __init__(self, d_model=256, seq_len=30):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Simple Residual Block with Conv/Linear (Simulating Mamba block for prototype)
        # In full version, import Mamba
        self.input_proj = nn.Linear(d_model, d_model)

        self.mid_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )

        self.output_proj = nn.Linear(d_model, d_model)


    def forward(self, x, t):
        # x: (batch, seq_len, d_model)
        # t: (batch,)

        t_emb = self.time_mlp(t) # (batch, d_model)
        t_emb = t_emb.unsqueeze(1) # (batch, 1, d_model)

        h = self.input_proj(x) + t_emb

        # Conv backbone (treating seq_len as spatial)
        # Permute for Conv1d: (batch, dim, seq)
        h = h.permute(0, 2, 1)
        h = self.mid_block(h)
        h = h.permute(0, 2, 1)

        out = self.output_proj(h)
        return out

class DiffusionGenerator:
    """
    DDPM Scheduler and Sampler
    """
    def __init__(self, model, timesteps=1000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device

        # Beta schedule
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: Add noise"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - self.alphas_cumprod[t])[:, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample(self, batch_size=4, seq_len=30, d_model=256):
        """Reverse diffusion: Generate from noise"""
        self.model.eval()
        with torch.no_grad():
            x = torch.randn((batch_size, seq_len, d_model)).to(self.device)

            for i in reversed(range(self.timesteps)):
                t = torch.tensor([i] * batch_size, device=self.device)

                # Predict noise
                predicted_noise = self.model(x, t)

                # Update x
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0

                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise

        return x

def test_diffusion():
    print("Initializing Diffusion Generator (MambaDiff)...")
    d_model = 64
    backbone = MambaDiff(d_model=d_model)
    diff = DiffusionGenerator(backbone, timesteps=50) # Fast test

    print("Running Generation Sample...")
    samples = diff.sample(batch_size=2, d_model=d_model)

    print(f"Generated Embedding Shape: {samples.shape}")
    # Verify values are not NaN
    if torch.isnan(samples).any():
        print("❌ Error: NaNs detected")
        sys.exit(1)
    else:
        print("✓ Generation successful (No NaNs)")

if __name__ == "__main__":
    test_diffusion()
