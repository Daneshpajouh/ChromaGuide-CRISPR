import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaMinimal(nn.Module):
    """
    A lightweight, pure-PyTorch implementation of the Mamba block.
    Use this when `mamba_ssm` (CUDA kernel) is not available.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Projects input to x and z (gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
        )

        # Projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + self.d_state * 2, bias=False) # Simplified rank

        # Only for implementation ease, we project to dt, B, C differently in full Mamba
        # But here we stick to a simplified parameterization for the fallback.

        # Let's align better with Mamba paper:
        # x -> (dt, B, C)
        self.dt_proj = nn.Linear(self.d_inner // 16, self.d_inner, bias=True) # dt_rank usually d_inner/16

        # We need specific projections for B and C
        self.x_proj_B = nn.Linear(self.d_inner, d_state, bias=False)
        self.x_proj_C = nn.Linear(self.d_inner, d_state, bias=False)
        self.x_proj_dt = nn.Linear(self.d_inner, self.d_inner, bias=True) # Simplified

        # S4D parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        x: (Batch, Seq_Len, Dim)
        """
        batch, seq_len, dim = x.shape

        x_and_z = self.in_proj(x) # (B, L, 2*D)
        x_in, z = x_and_z.chunk(2, dim=-1) # (B, L, D)

        # Conv1d
        x_conv = x_in.permute(0, 2, 1)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = self.act(x_conv) # (B, D, L)
        x_conv = x_conv.permute(0, 2, 1) # (B, L, D)

        # SSM (Selective Scan)
        # Simplified loop for pure torch

        # 1. Compute delta B C
        # In full mamba, delta comes from different projection
        # We'll use a simplified projection here for the fallback
        dt = F.softplus(self.x_proj_dt(x_conv)) # (B, L, D)
        B = self.x_proj_B(x_conv) # (B, L, N)
        C = self.x_proj_C(x_conv) # (B, L, N)

        # 2. Key parameters
        A = -torch.exp(self.A_log) # (D, N)

        # 3. Scan
        y = self.selective_scan(x_conv, dt, A, B, C)

        # Output
        out = y * self.act(z)
        out = self.out_proj(out)
        return out

    def selective_scan(self, u, dt, A, B, C):
        """
        Naive for-loop scan. Slow but correct.
        u: (B, L, D)
        dt: (B, L, D)
        A: (D, N)
        B: (B, L, N)
        C: (B, L, N)
        """
        batch, seq, d_inner = u.shape
        n = A.shape[1]

        # Discretize A
        # dA = exp(dt * A)
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A)) # (B, L, D, N)
        dB = torch.einsum('bld,bln->bldn', dt, B) # (B, L, D, N)

        h = torch.zeros(batch, d_inner, n, device=u.device)
        ys = []

        for i in range(seq):
            u_i = u[:, i, :] # (B, D)

            # h_t = dA * h_{t-1} + dB * u_t
            # dB * u: (B, D, N) * (B, D, 1) -> (B, D, N)

            # dB_u = dB[:, i, :, :] * u_i.unsqueeze(-1)

            # Using loop for clarity
            dA_i = dA[:, i, :, :] # (B, D, N)
            dB_i = dB[:, i, :, :] # (B, D, N)

            h = dA_i * h + dB_i * u_i.unsqueeze(-1)

            # y_t = C * h_t
            C_i = C[:, i, :] # (B, N)

            # (B, D, N) * (B, 1, N) -> sum over N
            y_i = torch.sum(h * C_i.unsqueeze(1), dim=-1) # (B, D)
            ys.append(y_i)

        return torch.stack(ys, dim=1) # (B, L, D)
