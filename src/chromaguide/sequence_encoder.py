"""Sequence encoder module for ChromaGuide.

Supports two architectures:
  (1) CNN-GRU baseline (ChromeCRISPR-style)
  (2) Mamba-based state-space model for O(L) linear complexity
  (3) DNABERT-2 transformer with k-mer tokenization

Inputs:
  - One-hot encoded sgRNA+PAM+protospacer sequence (4 channels x L positions)
  - Target-site sequence context

Outputs:
  - Sequence embedding z_s of shape (batch, d_model)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def kmer_tokenize(sequence: str, k: int = 6) -> List[int]:
    """Simple k-mer tokenizer for DNA sequences.

    Replaces AutoTokenizer from transformers to avoid PIL import issues.
    Converts sequence to overlapping k-mers, each mapped to a token ID.

    Args:
        sequence: DNA sequence string (e.g., "ACGTACGT")
        k: k-mer size (default 6 for DNABERT-2)

    Returns:
        List of token IDs (integers)
    """
    nt_to_id = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    tokens = []

    # Convert to k-mers
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        # Encode k-mer as a composite token
        token_id = 0
        for j, nt in enumerate(kmer):
            if nt in nt_to_id:
                token_id = token_id * 5 + nt_to_id[nt]
        tokens.append(token_id)

    return tokens if tokens else [0]  # Fallback to PAD token


def kmer_batch_tokenize(sequences: List[str], k: int = 6, max_len: int = 512) -> torch.Tensor:
    """Batch tokenize sequences to k-mers.

    Args:
        sequences: List of DNA sequences
        k: k-mer size
        max_len: Maximum sequence length (pads/truncates to this)

    Returns:
        Tensor of shape (batch_size, max_len) with token IDs
    """
    tokenized = [kmer_tokenize(seq, k) for seq in sequences]

    # Pad/truncate to max_len
    padded = []
    for tokens in tokenized:
        if len(tokens) >= max_len:
            padded.append(tokens[:max_len])
        else:
            padded.append(tokens + [0] * (max_len - len(tokens)))

    return torch.tensor(padded, dtype=torch.long)


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
    """DNABERT-2 transformer encoder with fallback to CNN when transformers unavailable.

    This encoder attempts to load DNABERT-2 weights, but gracefully falls back to
    a powerful CNN-based architecture if the transformers library has import issues
    (e.g., PIL circular imports on some Python versions).

    The fallback CNN still achieves competitive performance for on-target prediction.
    """
    def __init__(
        self,
        d_model: int = 64,  # PhD proposal constraint
        use_fallback: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__(d_model=d_model)

        self.d_model = d_model
        self.dropout_rate = dropout
        self.freeze_backbone = freeze_backbone

        # Try to load BERT, but fall back to CNN if import fails
        self.use_bert = False
        try:
            import json
            from pathlib import Path
            from transformers import BertConfig, BertModel

            cache_dir = self._resolve_dnabert_cache_dir(Path)
            config_path = None

            # Find config.json in cache snapshots
            if cache_dir is not None and cache_dir.exists():
                snapshots = sorted(cache_dir.glob("snapshots/*/config.json"))
                if snapshots:
                    config_path = snapshots[0]

            if config_path and config_path.exists():
                print(f"✓ Loading DNABERT-2 config from {config_path}", flush=True)
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = BertConfig(**config_dict)
            else:
                # Create default config for DNABERT-2
                print("Creating default DNABERT-2 config", flush=True)
                config = BertConfig(
                    vocab_size=4,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_act="gelu",
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    max_position_embeddings=512,
                    type_vocab_size=2,
                    initializer_range=0.02,
                    layer_norm_eps=1e-12,
                    pad_token_id=0,
                    position_embedding_type="relative_key_query",
                    use_cache=True,
                )

            # CRITICAL: Create model on CPU to avoid meta device tensors
            print("Creating DNABERT-2 model on CPU...", flush=True)
            self.backbone = BertModel(config, add_pooling_layer=False).to('cpu')

            # Load weights if available
            weights_path = None
            if cache_dir is not None and cache_dir.exists():
                weight_files = sorted(cache_dir.glob("snapshots/*/pytorch_model.bin"))
                if weight_files:
                    weights_path = weight_files[0]

            if weights_path and weights_path.exists():
                print(f"Loading DNABERT-2 weights from {weights_path}...", flush=True)
                try:
                    state_dict = torch.load(str(weights_path), map_location='cpu', weights_only=True)
                    self.backbone.load_state_dict(state_dict, strict=False)
                    print("✓ DNABERT-2 weights loaded", flush=True)
                except Exception as e:
                    print(f"⚠ Could not load weights: {e}", flush=True)

            # Project 768 -> d_model
            self.proj = nn.Sequential(
                nn.Linear(768, d_model),
                nn.LayerNorm(d_model),
            ) if d_model != 768 else nn.Identity()

            self.use_bert = True
            print("✓ DNABERT-2 mode: ENABLED", flush=True)

        except ImportError as e:
            if use_fallback:
                print(f"⚠ DNABERT-2 import failed ({str(e)[:60]}...), using CNN fallback", flush=True)
                # Build powerful CNN fallback
                self._init_cnn_fallback(d_model, dropout)
                self.use_bert = False
                print("✓ Using CNN fallback for sequence encoding", flush=True)
            else:
                raise

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _resolve_dnabert_cache_dir(path_cls):
        """Resolve DNABERT-2 HF cache path across local/HPC environments."""
        model_subdir = "models--zhihan1996--DNABERT-2-117M"
        candidates = []

        explicit = os.environ.get("DNABERT2_CACHE_DIR", "").strip()
        if explicit:
            candidates.append(path_cls(explicit).expanduser())

        hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
        if hf_hub_cache:
            base = path_cls(hf_hub_cache).expanduser()
            candidates.extend([base / model_subdir, base])

        hf_home = os.environ.get("HF_HOME", "").strip()
        if hf_home:
            base = path_cls(hf_home).expanduser()
            candidates.extend([base / "hub" / model_subdir, base / model_subdir])

        candidates.append(path_cls.home() / ".cache" / "huggingface" / "hub" / model_subdir)

        seen = set()
        for c in candidates:
            key = str(c)
            if key in seen:
                continue
            seen.add(key)
            if c.exists():
                return c

        # Fall back to conventional default path for clearer logging behavior.
        return path_cls.home() / ".cache" / "huggingface" / "hub" / model_subdir

    def _init_cnn_fallback(self, d_model: int, dropout: float):
        """Initialize CNN fallback architecture."""
        # Powerful conv-based encoder
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Token IDs (batch, L) OR one-hot (batch, 4, L)
        Returns:
            Sequence embedding (batch, d_model)
        """
        if self.use_bert:
            return self._forward_bert(x)
        else:
            return self._forward_cnn(x)

    def _forward_bert(self, x: torch.Tensor) -> torch.Tensor:
        """BERT forward path."""
        # Convert one-hot to tokens if needed
        if x.dim() == 3 and x.shape[1] == 4:
            x = x.argmax(dim=1)  # (batch, L)

        # Ensure on device
        device = next(self.backbone.parameters()).device
        x = x.to(device)

        # Forward; optionally freeze the heavy DNABERT backbone.
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(x)
        else:
            outputs = self.backbone(x)

        h = outputs[0].mean(dim=1)  # (batch, 768)
        h = self.proj(h)  # (batch, d_model)
        return self.dropout(h)

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """CNN fallback forward path."""
        # Convert tokens to one-hot if needed
        if x.dim() == 2 and x.dtype in [torch.long, torch.int64]:
            # Token IDs: convert to approximate one-hot
            batch, seq_len = x.shape
            one_hot = torch.zeros(batch, 4, seq_len, dtype=torch.float32, device=x.device)
            for b in range(batch):
                for i, token_id in enumerate(x[b]):
                    if 0 <= token_id < 4:
                        one_hot[b, token_id, i] = 1.0
            x = one_hot

        # CNN feature extraction
        h = self.conv_layers(x)  # (batch, 256, L')
        h = h.mean(dim=2)  # (batch, 256) - global avg pool
        h = self.fc(h)  # (batch, d_model)
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

        # Input projection from one-hot to d_model.
        # Use Linear over sequence positions instead of Conv1d(1x1) to avoid MPS backward issues.
        self.input_proj = nn.Linear(in_channels, d_model)
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
        # Project input: (batch, 4, L) -> (batch, L, 4) -> (batch, L, d_model)
        h = x.transpose(1, 2).contiguous()
        h = self.input_proj(h)
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
        x = self.norm(x).contiguous()

        # Split into two paths: SSM path and gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal convolution
        x_ssm = x_ssm.transpose(1, 2).contiguous()  # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :x.shape[1]]  # causal: trim to original length
        x_ssm = x_ssm.transpose(1, 2).contiguous()  # (B, L, d_inner)
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
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            x_t = x[:, t, :].contiguous()
            dt_t = dt[:, t, :].contiguous().unsqueeze(-1)  # (B, D, 1)
            A_bar = torch.exp(A.unsqueeze(0) * dt_t)  # (B, D, N)
            B_bar = dt_t * B_input[:, t, :].contiguous().unsqueeze(1)  # (B, D, N)

            h = A_bar * h + B_bar * x_t.unsqueeze(-1)  # (B, D, N)
            y_t = (h * C_input[:, t, :].contiguous().unsqueeze(1)).sum(-1)  # (B, D)
            outputs.append(y_t + self.D * x_t)

        return torch.stack(outputs, dim=1)
