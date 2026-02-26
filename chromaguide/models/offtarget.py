"""Off-target prediction module.

Module 2 of ChromaGuide: predicts off-target cleavage probability
for candidate sgRNAs at genome-wide off-target sites.

Architecture:
    1. Per-site CNN scorer:
       - Input: Encoded guide-target alignment (mismatches + bulges + chromatin)
       - Output: p(cleavage | site)
    
    2. Noisy-OR aggregation:
       - Aggregates per-site cleavage probabilities
       - P(off-target | guide) = 1 - Π(1 - p_i)  (Noisy-OR)
    
Performance target: AUROC ≥ 0.92 (vs CCLMoff baseline 0.81)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Guide-Target Pair Encoding
# ═══════════════════════════════════════════════════════════════

def encode_guide_target_pair(
    guide: str,
    target: str,
    max_len: int = 23,
) -> torch.Tensor:
    """Encode a guide-target alignment into a feature tensor.
    
    Encoding channels:
        0-3:   Guide one-hot (A, C, G, T)
        4-7:   Target one-hot (A, C, G, T)
        8:     Match indicator (1 if guide[i] == target[i])
        9:     Mismatch indicator (1 if guide[i] != target[i])
        10:    Bulge indicator (placeholder)
        11:    Position encoding (i / max_len)
    
    Returns:
        Tensor of shape (12, max_len).
    """
    DNA = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoding = torch.zeros(12, max_len)
    
    guide = guide.upper()[:max_len]
    target = target.upper()[:max_len]
    
    for i, (g, t) in enumerate(zip(guide, target)):
        # Guide one-hot
        if g in DNA:
            encoding[DNA[g], i] = 1.0
        # Target one-hot
        if t in DNA:
            encoding[4 + DNA[t], i] = 1.0
        # Match / mismatch
        if g == t:
            encoding[8, i] = 1.0
        else:
            encoding[9, i] = 1.0
        # Position encoding
        encoding[11, i] = i / max_len
    
    return encoding


def batch_encode_pairs(
    guides: list[str],
    targets: list[str],
    max_len: int = 23,
) -> torch.Tensor:
    """Batch encode guide-target pairs.
    
    Returns:
        Tensor of shape (batch, 12, max_len).
    """
    return torch.stack([
        encode_guide_target_pair(g, t, max_len)
        for g, t in zip(guides, targets)
    ])


# ═══════════════════════════════════════════════════════════════
# Per-site CNN Scorer
# ═══════════════════════════════════════════════════════════════

class OffTargetSiteScorerCNN(nn.Module):
    """CNN-based scorer for individual guide-target site pairs.
    
    Input: Encoded guide-target alignment (batch, 12, seq_len)
    Optional: Chromatin features at the off-target site
    Output: p(cleavage | guide, target_site) ∈ (0, 1)
    """
    
    def __init__(
        self,
        input_channels: int = 12,
        kernel_sizes: list[int] = [3, 5, 7],
        n_filters: int = 64,
        fc_dims: list[int] = [128, 64],
        use_chromatin: bool = True,
        chromatin_dim: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Multi-scale CNN
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, n_filters, k, padding=k // 2),
                nn.BatchNorm1d(n_filters),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        
        cnn_out = n_filters * len(kernel_sizes)
        
        # Global pooling
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Optional chromatin features
        self.use_chromatin = use_chromatin
        fc_input = cnn_out + (chromatin_dim if use_chromatin else 0)
        
        # FC layers
        layers = []
        in_dim = fc_input
        for h_dim in fc_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.fc = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        chromatin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Encoded guide-target pair (batch, 12, seq_len).
            chromatin: Chromatin features at off-target site (batch, chromatin_dim).
        
        Returns:
            Cleavage probability (batch, 1).
        """
        # Multi-scale CNN
        conv_outs = [conv(x) for conv in self.convs]
        h = torch.cat(conv_outs, dim=1)  # (batch, cnn_out, seq_len)
        h = self.pool(h).squeeze(-1)     # (batch, cnn_out)
        
        # Concatenate chromatin features
        if self.use_chromatin and chromatin is not None:
            h = torch.cat([h, chromatin], dim=-1)
        
        # Predict cleavage probability
        logit = self.fc(h)  # (batch, 1)
        return torch.sigmoid(logit)


# ═══════════════════════════════════════════════════════════════
# Noisy-OR Aggregation
# ═══════════════════════════════════════════════════════════════

class NoisyORAggregation(nn.Module):
    """Noisy-OR aggregation for genome-wide off-target risk.
    
    Given per-site cleavage probabilities p_1, ..., p_K:
        P(any off-target) = 1 - Π_{i=1}^{K} (1 - p_i)
    
    Optionally uses only top-K sites to focus on highest-risk loci.
    Temperature scaling can sharpen/smooth the aggregation.
    """
    
    def __init__(
        self,
        top_k: int = 20,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("log_temperature", torch.tensor(float(temperature)).log())
    
    @property
    def temperature(self):
        return self.log_temperature.exp()
    
    def forward(self, site_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            site_probs: Per-site cleavage probabilities (batch, n_sites).
        
        Returns:
            Aggregated off-target risk score (batch, 1).
        """
        # Temperature scaling
        scaled = site_probs ** (1.0 / self.temperature)
        
        # Select top-K most risky sites
        if self.top_k > 0 and scaled.shape[1] > self.top_k:
            topk_vals, _ = torch.topk(scaled, self.top_k, dim=1)
        else:
            topk_vals = scaled
        
        # Noisy-OR: 1 - Π(1 - p_i)
        # Use log-space for numerical stability: log(Π(1-p)) = Σlog(1-p)
        log_survival = torch.log1p(-topk_vals.clamp(max=1.0 - 1e-7))
        risk = 1.0 - torch.exp(log_survival.sum(dim=1, keepdim=True))
        
        return risk


# ═══════════════════════════════════════════════════════════════
# Full Off-Target Model
# ═══════════════════════════════════════════════════════════════

class OffTargetModel(nn.Module):
    """Complete off-target prediction pipeline.
    
    1. Score each candidate off-target site with CNN scorer
    2. Aggregate via Noisy-OR
    3. Output: P(off-target activity | sgRNA)
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.site_scorer = OffTargetSiteScorerCNN(
            kernel_sizes=list(cfg.offtarget_model.scoring.kernel_sizes),
            n_filters=cfg.offtarget_model.scoring.n_filters,
            fc_dims=list(cfg.offtarget_model.scoring.fc_dims),
            use_chromatin=cfg.offtarget_model.scoring.use_chromatin,
        )
        
        self.aggregation = NoisyORAggregation(
            top_k=cfg.offtarget_model.aggregation.top_k,
            temperature=cfg.offtarget_model.aggregation.temperature,
        )
    
    def score_sites(
        self,
        pairs: torch.Tensor,
        chromatin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score individual off-target sites.
        
        Args:
            pairs: Encoded guide-target pairs (batch * n_sites, 12, seq_len).
            chromatin: Chromatin at off-target sites (batch * n_sites, 3).
        
        Returns:
            Per-site scores (batch * n_sites, 1).
        """
        return self.site_scorer(pairs, chromatin)
    
    def forward(
        self,
        pairs: torch.Tensor,
        chromatin: torch.Tensor | None = None,
        n_sites_per_guide: int = 20,
    ) -> torch.Tensor:
        """Full off-target prediction.
        
        Args:
            pairs: Encoded pairs, flattened (batch * n_sites, 12, seq_len).
            chromatin: Chromatin features (batch * n_sites, 3).
            n_sites_per_guide: Number of off-target sites per guide.
        
        Returns:
            Off-target risk score (batch, 1).
        """
        # Score all sites
        site_scores = self.site_scorer(pairs, chromatin).squeeze(-1)  # (batch * n_sites,)
        
        # Reshape to (batch, n_sites)
        batch_size = site_scores.shape[0] // n_sites_per_guide
        site_scores = site_scores.view(batch_size, n_sites_per_guide)
        
        # Aggregate
        risk = self.aggregation(site_scores)  # (batch, 1)
        
        return risk
