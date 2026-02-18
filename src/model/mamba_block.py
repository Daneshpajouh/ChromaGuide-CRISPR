import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint

# Try importing the official CUDA kernel
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("SafeMamba: CUDA Mamba kernels available.")
except ImportError:
    MAMBA_AVAILABLE = False
    print("SafeMamba: mamba_ssm not found. Using PyTorch fallback (MPS-Optimized).")

# @torch.jit.script
def selective_scan_loop(u: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    JIT-Compiled Selective Scan Loop.
    Optimized for MPS by fusing kernel launches.

    Args:
        u: (B, L, D_inner)
        dt: (B, L, D_inner)
        A: (D_inner, D_state)
        B: (B, L, D_state)
        C: (B, L, D_state)
        D: (D_inner)
    """
    batch, seq, d_inner = u.shape
    d_state = A.shape[1]

    # Precompute Delta terms
    dt_ex = dt.unsqueeze(-1) # (B, L, D, 1)
    dA = torch.exp(dt_ex * A) # (B, L, D, N)
    dB_u = (dt_ex * B.unsqueeze(2)) * u.unsqueeze(-1) # (B, L, D, N)

    h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []

    for t in range(seq):
        # Scan step: h = dA * h + dB_u
        h = dA[:, t] * h + dB_u[:, t]

        # Output step: y = sum(h * C)
        # C[:, t]: (B, N) -> (B, 1, N) broadcast
        y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=1) # (B, L, D)
    y = y + u * D
    return y

class SafeMambaBlock(nn.Module):
    """
    Hardware-Aware Mamba Block.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model

        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
        else:
            self.mamba = MambaPyTorch(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + shortcut

class MambaPyTorch(nn.Module):
    """
    A pure PyTorch implementation of the Selective State Space Model (S6).
    Optimized for Apple Silicon (MPS).
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        x = self.act(x)

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        # Call JIT-compiled loop
        # Use Gradient Checkpointing to save memory
        # FORCE FP32 for Scan Stability on MPS
        x_f32 = x.float()
        dt_f32 = dt.float()
        A_log_f32 = self.A_log.float()
        B_f32 = B.float()
        C_f32 = C.float()
        D_f32 = self.D.float()

        if self.training:
            y = torch.utils.checkpoint.checkpoint(
                selective_scan_loop, x_f32, dt_f32, -torch.exp(A_log_f32), B_f32, C_f32, D_f32,
                use_reentrant=False
            )
        else:
            y = selective_scan_loop(x_f32, dt_f32, -torch.exp(A_log_f32), B_f32, C_f32, D_f32)

        y = y.to(dtype=self.out_proj.weight.dtype) # Cast back to model dtype (FP16/BF16) if needed
        y = y * self.act(z)
        return self.out_proj(y)
