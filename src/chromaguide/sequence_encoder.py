"""Sequence encoder module for ChromaGuide.

Supports two architectures:
  (1) CNN-GRU baseline (ChromeCRISPR-style)
  (2) Mamba-based state-space model for O(L) linear complexity

Inputs:
  - One-hot encoded sgRNA+PAM+protospacer sequence (4 channels x L positions)
  - Target-site sequence context

Outputs:
  - Sequence embedding z_s of shape (batch, d_model)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SequenceEncoder(nn.Module):
    """Abstract base for sequence encoders in ChromaGuide."""
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CNNGRUEncoder(SequenceEncoder):
    """ChromeCRISPR-style CNN-GRU hybrid encoder.

    Architecture:
      Conv1D layers -> Bidirectional GRU -> Global pooling -> Linear projection

    Processes one-hot encoded 23-nt sgRNA+PAM sequence (4 channels x 23 positions)
    plus optional flanking context.
    """
    def __init__(
        self,
        in_channels: int = 4,
        seq_len: int = 21,
        n_filters: list = None,
        kernel_sizes: list = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        d_model: int = 256,  # UPDATED: Matches PhD proposal requirement (zs in R^256)
        dropout: float = 0.3,
    ):
        super().__init__(d_model=d_model)
        if n_filters is None:
            n_filters = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        # Multi-scale CNN feature extraction
        self.conv_branches = nn.ModuleList()
        for nf, ks in zip(n_filters, kernel_sizes):
            branch = nn.Sequential(
                nn.Conv1d(in_channels, nf, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(nf),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.conv_branches.append(branch)

        total_filters = sum(n_filters)

        # Bidirectional GRU for sequential dependency capture
        self.gru = nn.GRU(
            input_size=total_filters,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Project to d_model
        self.proj = nn.Sequential(
            nn.Linear(gru_hidden * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: One-hot encoded sequence (batch, 4, seq_len)

        Returns:
            Sequence embedding (batch, d_model)
        """
        # Multi-scale CNN
        branch_outputs = [branch(x) for branch in self.conv_branches]
        h = torch.cat(branch_outputs, dim=1)  # (batch, total_filters, seq_len)

        # GRU expects (batch, seq_len, features)
        h = h.transpose(1, 2)
        h, _ = self.gru(h)  # (batch, seq_len, 2*gru_hidden)

        # Global average pooling over sequence length
        h = h.mean(dim=1)  # (batch, 2*gru_hidden)

        return self.proj(h)  # (batch, d_model)


class DNABERT2Encoder(SequenceEncoder):
    """DNABERT-2 transformer encoder.

    Uses a pre-trained transformer to extract features from gRNA sequences.
    Optimized for short-range gRNA context and PAM recognition.
    """
    def __init__(
        self,
        d_model: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__(d_model=d_model)
        from transformers import AutoModel, AutoConfig

        # Load pre-trained DNABERT-2 with custom config fix
        config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = 0 # Default for DNABERT-2

        # PREVENT "meta" device errors: Ensure Alibi tensors are on CPU during init
        # We use a context manager to force CPU initialization for DNABERT-2 as a workaround
        # for a known issue with the model's Alibi implementation on GPU nodes.
        import torch
        from transformers import AutoModel, AutoConfig

        # Force CPU device for the duration of model loading
        with torch.device('cpu'):
            self.backbone = AutoModel.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                config=config,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                device_map=None
            )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tokenized sequence IDs (batch, L)
        Returns:
            Sequence embedding (batch, d_model)
        """
        # Multi-hot/One-hot check (DNABERT-2 expects token IDs)
        if x.dim() == 3:
            # Emergency fallback: convert one-hot to approximate representation
            # or throw error. For now, we project if training loop hasn't been updated.
            x = x.argmax(dim=1)

        outputs = self.backbone(x)
        # Global pooling (using mean pooling for gRNAs)
        h = outputs[0].mean(dim=1)
        return self.dropout(h)


class MambaSequenceEncoder(SequenceEncoder):
    """Mamba-based state-space model encoder.

    Uses selective state space model (S6) for O(L) linear complexity
    with input-dependent gating, achieving long-range dependency capture
    without quadratic attention cost.

    Architecture:
      Embedding -> N x MambaBlock -> LayerNorm -> Global pooling -> Linear
    """
    def __init__(
        self,
        in_channels: int = 4,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(d_model=d_model)

        # Input projection from one-hot to d_model
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=1),
            nn.LayerNorm(d_model) if False else nn.Identity(),  # applied after transpose
        )
        self.input_norm = nn.LayerNorm(d_model)

        # Build Mamba layers (simplified SSM blocks)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
            )

        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: One-hot encoded sequence (batch, 4, seq_len)

        Returns:
            Sequence embedding (batch, d_model)
        """
        # Project input: (batch, 4, L) -> (batch, d_model, L) -> (batch, L, d_model)
        h = self.input_proj(x).transpose(1, 2)
        h = self.input_norm(h)

        # Mamba layers with residual connections
        for layer in self.layers:
            h = layer(h)

        h = self.norm_f(h)

        # Global average pooling
        return h.mean(dim=1)  # (batch, d_model)


class MambaBlock(nn.Module):
    """Simplified Mamba (S6) block with selective gating.

    Implements the core selective state-space mechanism with:
      - Input-dependent selection mechanism (Delta, B, C projections)
      - Causal 1D convolution for local context
      - Gated output with SiLU activation
    """
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.norm = nn.LayerNorm(d_model)

        # Input projection (expand)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal 1D convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters: dt, B, C projections from input
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # Learnable A (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Split into two paths: SSM path and gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal convolution
        x_ssm = x_ssm.transpose(1, 2)  # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :x.shape[1]]  # causal: trim to original length
        x_ssm = x_ssm.transpose(1, 2)  # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # Selective SSM
        y = self._selective_ssm(x_ssm)

        # Gated output
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y

    def _selective_ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective state-space model computation.

        This implements the discretized SSM with input-dependent
        parameters (selective mechanism).
        """
        B, L, D = x.shape

        # Input-dependent projections
        dt = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        B_input = self.B_proj(x)  # (B, L, d_state)
        C_input = self.C_proj(x)  # (B, L, d_state)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Sequential scan (can be parallelized with associative scan)
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, D, 1)
            A_bar = torch.exp(A.unsqueeze(0) * dt_t)  # (B, D, N)
            B_bar = dt_t * B_input[:, t, :].unsqueeze(1)  # (B, D, N)

            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)  # (B, D, N)
            y_t = (h * C_input[:, t, :].unsqueeze(1)).sum(-1)  # (B, D)
            y[:, t, :] = y_t + self.D * x[:, t, :]

        return y
