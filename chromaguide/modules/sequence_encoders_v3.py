"""Enhanced sequence encoders for ChromaGuide v3.

Key improvements over v2:
    1. Multi-Scale CNN with kernels [1, 3, 5, 7] (from CRISPR-FMC SOTA)
    2. Transformer encoder layer for long-range dependencies
    3. Dual-branch encoding: one-hot + learned embedding (from CrnnCrispr)
    4. Handcrafted features: GC content, position-specific encoding
    5. Wider model capacity: 128 filters, 256 GRU hidden
    6. Residual connections throughout
    7. Fixed transformer backbone integration with proper error handling

References:
    - CRISPR-FMC (Li et al. 2025): MSC-CNN + Transformer + BiGRU
    - CrnnCrispr (Zhu et al. 2024): Dual CNN + BiGRU
    - DeepMEns (Jia et al. 2024): 5-model ensemble with diverse features
"""
from __future__ import annotations

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)

DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


# ═══════════════════════════════════════════════════════════════
# Handcrafted Feature Extractor
# ═══════════════════════════════════════════════════════════════

class HandcraftedFeatures(nn.Module):
    """Extract biologically relevant handcrafted features from one-hot sequence.
    
    Features (computed from one-hot tensor, no raw strings needed):
        - GC content (scalar)
        - Position-specific nucleotide frequencies (4 features)
        - Dinucleotide frequencies at PAM-proximal/distal regions (8 features)
        - PAM sequence encoding (3 features)
    
    Total: 16 features
    """
    
    def __init__(self, seq_len: int = 23):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = 16
        # Learnable projection
        self.fc = nn.Sequential(
            nn.Linear(self.n_features, 32),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.output_dim = 32
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: One-hot encoded DNA (batch, 4, seq_len)
        Returns:
            Feature vector (batch, 32)
        """
        batch_size = x.shape[0]
        device = x.device
        features = []
        
        # GC content
        gc = (x[:, 1, :] + x[:, 2, :]).sum(dim=1, keepdim=True) / self.seq_len  # (batch, 1)
        features.append(gc)
        
        # Position-weighted nucleotide content (PAM-proximal vs distal)
        # PAM is last 3 positions (positions 20-22 for 23-nt)
        pam_proximal = x[:, :, -6:]  # last 6 positions
        pam_distal = x[:, :, :6]     # first 6 positions
        
        prox_content = pam_proximal.mean(dim=2)  # (batch, 4)
        dist_content = pam_distal.mean(dim=2)    # (batch, 4)
        features.append(prox_content)
        features.append(dist_content)
        
        # Dinucleotide content at key positions
        # AA, AC, AG, AT frequencies in seed region (pos 1-12)
        seed_region = x[:, :, :12]
        # Simplified: use product of adjacent positions
        dinuc_feat = (seed_region[:, :, :-1] * seed_region[:, :, 1:]).sum(dim=2) / 11.0  # (batch, 4)
        features.append(dinuc_feat)
        
        # PAM encoding (positions 20-22, expecting NGG)
        pam = x[:, :, -3:]  # (batch, 4, 3)
        pam_flat = pam.mean(dim=1)  # (batch, 3)
        features.append(pam_flat)
        
        features = torch.cat(features, dim=1)  # (batch, 16)
        return self.fc(features)


# ═══════════════════════════════════════════════════════════════
# Multi-Scale Convolutional Module (from CRISPR-FMC)
# ═══════════════════════════════════════════════════════════════

class MultiScaleCNN(nn.Module):
    """Multi-Scale Convolutional module with 4 parallel kernel sizes.
    
    Inspired by CRISPR-FMC's MSC module (Li et al. 2025).
    Uses kernel sizes [1, 3, 5, 7] to capture motifs at different scales.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        n_filters: int = 128,
        kernel_sizes: list[int] = [1, 3, 5, 7],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, n_filters, k, padding=k // 2),
                nn.BatchNorm1d(n_filters),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        
        # Compress multi-scale features
        total_filters = n_filters * len(kernel_sizes)
        self.compress = nn.Sequential(
            nn.Conv1d(total_filters, n_filters * 2, 1),
            nn.BatchNorm1d(n_filters * 2),
            nn.GELU(),
        )
        
        self.output_channels = n_filters * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_channels, seq_len)
        Returns:
            (batch, output_channels, seq_len)
        """
        conv_outs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        x = self.compress(x)
        return x


# ═══════════════════════════════════════════════════════════════
# Enhanced CNN-GRU Encoder v3
# ═══════════════════════════════════════════════════════════════

class CNNGRUEncoderV3(nn.Module):
    """Enhanced CNN-GRU encoder (v3) incorporating SOTA innovations.
    
    Architecture (inspired by CRISPR-FMC + CrnnCrispr + DeepMEns):
        Branch 1: One-hot → Multi-Scale CNN [1,3,5,7] → Transformer → BiGRU → Attention Pool
        Branch 2: One-hot → Learned Embedding → BiGRU → Attention Pool
        Handcrafted features: GC content, position-specific, PAM
        
        → Concatenate all branches → MLP → z_s
    
    Improvements over v2 CNNGRUEncoder:
        - Multi-scale CNN (4 kernel sizes vs 3)
        - Transformer self-attention layer
        - Dual branch (one-hot + embedding)
        - Handcrafted biological features
        - Wider model (128 filters, 256 GRU)
        - Residual connections
        - Better attention pooling
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        kernel_sizes: list[int] = [1, 3, 5, 7],
        n_filters: int = 128,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        output_dim: int = 128,
        n_transformer_heads: int = 4,
        n_transformer_layers: int = 1,
        seq_len: int = 23,
        cnn_dropout: float = 0.1,
        gru_dropout: float = 0.2,
        embedding_dim: int = 64,
        vocab_size: int = 5,  # A,C,G,T,N
    ):
        super().__init__()
        self.output_dim = output_dim
        
        # ---- Branch 1: Multi-Scale CNN + Transformer + BiGRU ----
        self.msc = MultiScaleCNN(
            input_channels=input_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            dropout=cnn_dropout,
        )
        
        cnn_out_dim = self.msc.output_channels  # 256
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_dim,
            nhead=n_transformer_heads,
            dim_feedforward=cnn_out_dim * 2,
            dropout=cnn_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )
        
        # BiGRU
        self.gru1 = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        gru1_out_dim = gru_hidden * 2
        
        # Attention pooling for Branch 1
        self.attn1 = nn.Sequential(
            nn.Linear(gru1_out_dim, gru1_out_dim // 2),
            nn.Tanh(),
            nn.Linear(gru1_out_dim // 2, 1),
        )
        
        # ---- Branch 2: Learned Embedding + BiGRU ----
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embedding_dim) * 0.02)
        self.emb_dropout = nn.Dropout(0.1)
        
        self.gru2 = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        gru2_out_dim = gru_hidden  # // 2 * 2 = gru_hidden
        
        # Attention pooling for Branch 2
        self.attn2 = nn.Sequential(
            nn.Linear(gru2_out_dim, gru2_out_dim // 2),
            nn.Tanh(),
            nn.Linear(gru2_out_dim // 2, 1),
        )
        
        # ---- Branch 3: Handcrafted features ----
        self.handcrafted = HandcraftedFeatures(seq_len=seq_len)
        
        # ---- Fusion ----
        total_dim = gru1_out_dim + gru2_out_dim + self.handcrafted.output_dim
        
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(gru_dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(gru_dropout / 2),
        )
    
    @staticmethod
    def _attention_pool(h: torch.Tensor, attn_net: nn.Module) -> torch.Tensor:
        """Attention-weighted pooling."""
        attn_weights = attn_net(h)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        return (h * attn_weights).sum(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: One-hot encoded DNA (batch, 4, seq_len)
        Returns:
            z_s of shape (batch, output_dim)
        """
        # Branch 1: MSC-CNN + Transformer + BiGRU
        h1 = self.msc(x)  # (batch, cnn_out, seq_len)
        h1 = h1.permute(0, 2, 1)  # (batch, seq_len, cnn_out)
        h1 = self.transformer(h1)  # (batch, seq_len, cnn_out)
        h1, _ = self.gru1(h1)  # (batch, seq_len, gru1_out)
        z1 = self._attention_pool(h1, self.attn1)  # (batch, gru1_out)
        
        # Branch 2: Embedding + BiGRU
        tokens = x.argmax(dim=1)  # (batch, seq_len) from one-hot
        h2 = self.embedding(tokens) + self.pos_embedding[:, :x.shape[2], :]
        h2 = self.emb_dropout(h2)
        h2, _ = self.gru2(h2)  # (batch, seq_len, gru2_out)
        z2 = self._attention_pool(h2, self.attn2)  # (batch, gru2_out)
        
        # Branch 3: Handcrafted features
        z3 = self.handcrafted(x)  # (batch, 32)
        
        # Fusion
        z = torch.cat([z1, z2, z3], dim=-1)
        z_s = self.projection(z)
        
        return z_s


# ═══════════════════════════════════════════════════════════════
# Enhanced Caduceus Encoder v3
# ═══════════════════════════════════════════════════════════════

class CaduceusEncoderV3(nn.Module):
    """Enhanced Caduceus encoder with multi-scale features.
    
    Improvements over v2:
        - Added multi-scale CNN branch alongside Mamba
        - Deeper architecture (6 layers vs 4)
        - Handcrafted features integration
        - Better RC-equivariant pooling
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        output_dim: int = 128,
        vocab_size: int = 5,
        max_len: int = 23,
        dropout: float = 0.1,
        rc_equivariant: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.d_model = d_model
        self.rc_equivariant = rc_equivariant
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.emb_dropout = nn.Dropout(dropout)
        
        # BiMamba layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                BiMambaBlockV3(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand_factor=expand_factor,
                    dropout=dropout,
                )
            )
        
        self.norm = nn.LayerNorm(d_model)
        
        # Multi-scale CNN branch
        self.cnn_branch = MultiScaleCNN(
            input_channels=4,
            n_filters=64,
            kernel_sizes=[1, 3, 5, 7],
            dropout=dropout,
        )
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        
        # Handcrafted features
        self.handcrafted = HandcraftedFeatures(seq_len=max_len)
        
        # Projection: Mamba + CNN + Handcrafted → output
        total_dim = d_model + self.cnn_branch.output_channels + self.handcrafted.output_dim
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
    
    @staticmethod
    def rc_complement_tokens(tokens: torch.Tensor) -> torch.Tensor:
        rc_map = torch.tensor([3, 2, 1, 0, 4], device=tokens.device)
        rc_tokens = rc_map[tokens]
        return rc_tokens.flip(dims=[-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: One-hot (batch, 4, seq_len) or token indices (batch, seq_len)
        Returns:
            z_s of shape (batch, output_dim)
        """
        # Convert one-hot to tokens
        if x.dim() == 3 and x.shape[1] == 4:
            tokens = x.argmax(dim=1)
            one_hot = x
        else:
            tokens = x
            one_hot = None
        
        # Mamba branch
        h = self.embedding(tokens) + self.pos_encoding[:, :tokens.shape[1], :]
        h = self.emb_dropout(h)
        
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        
        if self.rc_equivariant:
            rc_tokens = self.rc_complement_tokens(tokens)
            h_rc = self.embedding(rc_tokens) + self.pos_encoding[:, :rc_tokens.shape[1], :]
            h_rc = self.emb_dropout(h_rc)
            for layer in self.layers:
                h_rc = layer(h_rc)
            h_rc = self.norm(h_rc)
            z_mamba = (h.mean(dim=1) + h_rc.mean(dim=1)) / 2
        else:
            z_mamba = h.mean(dim=1)
        
        # CNN branch
        if one_hot is not None:
            cnn_out = self.cnn_branch(one_hot)  # (batch, channels, seq_len)
            z_cnn = self.cnn_pool(cnn_out).squeeze(-1)  # (batch, channels)
        else:
            z_cnn = torch.zeros(tokens.shape[0], self.cnn_branch.output_channels, device=tokens.device)
        
        # Handcrafted features
        if one_hot is not None:
            z_hand = self.handcrafted(one_hot)
        else:
            z_hand = torch.zeros(tokens.shape[0], self.handcrafted.output_dim, device=tokens.device)
        
        # Combine
        z = torch.cat([z_mamba, z_cnn, z_hand], dim=-1)
        return self.projection(z)


class BiMambaBlockV3(nn.Module):
    """Enhanced BiMamba block with better residual connections."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        
        self._use_mamba = False
        try:
            from mamba_ssm import Mamba
            self.fwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand_factor)
            self.rev_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand_factor)
            self._use_mamba = True
        except ImportError:
            self.fwd_rnn = nn.GRU(d_model, d_model, batch_first=True)
            self.rev_rnn = nn.GRU(d_model, d_model, batch_first=True)
        
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN for post-processing (like Transformer)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        if self._use_mamba:
            fwd_out = self.fwd_mamba(x)
            rev_out = self.rev_mamba(x.flip(dims=[1])).flip(dims=[1])
        else:
            fwd_out, _ = self.fwd_rnn(x)
            rev_out, _ = self.rev_rnn(x.flip(dims=[1]))
            rev_out = rev_out.flip(dims=[1])
        
        gate = self.gate(torch.cat([fwd_out, rev_out], dim=-1))
        combined = gate * fwd_out + (1 - gate) * rev_out
        
        out = self.out_proj(combined)
        x = out + residual  # First residual
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


# ═══════════════════════════════════════════════════════════════
# Fixed DNABERT-2 Encoder v3
# ═══════════════════════════════════════════════════════════════

class DNABERT2EncoderV3(nn.Module):
    """Fixed DNABERT-2 encoder with proper error handling and fallback.
    
    Key fixes over v2:
        - Explicit logging on model load failure (no silent pass)
        - Trainable lightweight transformer fallback when pretrained model unavailable
        - Differential learning rates support
        - Proper gradient flow through frozen layers
    """
    
    def __init__(
        self,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        output_dim: int = 128,
        freeze_layers: int = 8,  # Freeze first 8 of 12 layers
        dropout: float = 0.1,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self._hidden_dim = 768
        self._pretrained_loaded = False
        
        try:
            from transformers import AutoModel, AutoTokenizer
            logger.info(f"Loading DNABERT-2 from {model_name}...")
            kwargs = {"trust_remote_code": True}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            self.backbone = AutoModel.from_pretrained(model_name, **kwargs)
            self._hidden_dim = self.backbone.config.hidden_size
            self._pretrained_loaded = True
            logger.info(f"DNABERT-2 loaded successfully. Hidden dim: {self._hidden_dim}")
            
            # Freeze lower layers
            if freeze_layers > 0:
                frozen_count = 0
                for name, param in self.backbone.named_parameters():
                    should_freeze = False
                    if "embeddings" in name:
                        should_freeze = True
                    for i in range(freeze_layers):
                        if f"layer.{i}." in name or f"encoder.layer.{i}." in name:
                            should_freeze = True
                    if should_freeze:
                        param.requires_grad = False
                        frozen_count += 1
                logger.info(f"Frozen {frozen_count} parameters in first {freeze_layers} layers")
                
        except Exception as e:
            logger.error(f"FAILED to load DNABERT-2: {e}")
            logger.info("Falling back to trainable lightweight transformer")
            self.tokenizer = None
            self.backbone = None
            self._build_fallback_transformer()
        
        # Projection head with MLP
        self.projection = nn.Sequential(
            nn.Linear(self._hidden_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        
        # Handcrafted features (always available)
        self.handcrafted = HandcraftedFeatures(seq_len=23)
        self.feature_combine = nn.Linear(output_dim + self.handcrafted.output_dim, output_dim)
    
    def _build_fallback_transformer(self):
        """Build a trainable lightweight transformer as fallback."""
        self._hidden_dim = 256
        vocab_size = 5
        max_len = 23
        
        self.fallback_embedding = nn.Embedding(vocab_size, self._hidden_dim)
        self.fallback_pos = nn.Parameter(torch.randn(1, max_len, self._hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._hidden_dim,
            nhead=4,
            dim_feedforward=self._hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.fallback_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fallback_norm = nn.LayerNorm(self._hidden_dim)
        
        logger.info(f"Built fallback transformer with {sum(p.numel() for p in self.parameters()):,} params")
    
    def forward(self, x: torch.Tensor, sequences: list[str] | None = None) -> torch.Tensor:
        """
        Args:
            x: One-hot tensor (batch, 4, seq_len)
            sequences: Raw DNA strings for tokenization (used with pretrained model)
        Returns:
            z_s of shape (batch, output_dim)
        """
        if self._pretrained_loaded and sequences is not None:
            # Use pretrained DNABERT-2
            tokens = self.tokenizer(
                sequences, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(next(self.backbone.parameters()).device)
            outputs = self.backbone(**tokens)
            # Mean pooling (more robust than CLS for short sequences)
            h = outputs.last_hidden_state.mean(dim=1)
        elif hasattr(self, 'fallback_transformer'):
            # Fallback transformer
            tokens = x.argmax(dim=1)  # (batch, seq_len)
            h = self.fallback_embedding(tokens) + self.fallback_pos[:, :tokens.shape[1], :]
            h = self.fallback_transformer(h)
            h = self.fallback_norm(h)
            h = h.mean(dim=1)
        else:
            # Last resort: use one-hot features only
            h = x.mean(dim=2)  # (batch, 4)
            h = F.pad(h, (0, self._hidden_dim - 4))
        
        z_backbone = self.projection(h)
        
        # Add handcrafted features
        z_hand = self.handcrafted(x)
        z_s = self.feature_combine(torch.cat([z_backbone, z_hand], dim=-1))
        
        return z_s


# ═══════════════════════════════════════════════════════════════
# Fixed Evo Encoder v3
# ═══════════════════════════════════════════════════════════════

class EvoEncoderV3(nn.Module):
    """Fixed Evo encoder with proper error handling and fallback.
    
    Same fix pattern as DNABERT2EncoderV3.
    """
    
    def __init__(
        self,
        model_name: str = "togethercomputer/evo-1-131k-base",
        output_dim: int = 128,
        lora_r: int = 16,
        lora_alpha: int = 32,
        dropout: float = 0.1,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self._hidden_dim = 512
        self._pretrained_loaded = False
        
        try:
            from transformers import AutoModel, AutoTokenizer
            logger.info(f"Loading Evo from {model_name}...")
            kwargs = {"trust_remote_code": True}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            self.backbone = AutoModel.from_pretrained(model_name, **kwargs)
            
            if hasattr(self.backbone.config, "hidden_size"):
                self._hidden_dim = self.backbone.config.hidden_size
            elif hasattr(self.backbone.config, "d_model"):
                self._hidden_dim = self.backbone.config.d_model
            
            # Freeze backbone, add LoRA
            for param in self.backbone.parameters():
                param.requires_grad = False
            self._apply_lora(lora_r, lora_alpha)
            self._pretrained_loaded = True
            logger.info(f"Evo loaded. Hidden dim: {self._hidden_dim}")
            
        except Exception as e:
            logger.error(f"FAILED to load Evo: {e}")
            logger.info("Falling back to trainable Mamba-style encoder")
            self.tokenizer = None
            self.backbone = None
            self._build_fallback()
        
        self.projection = nn.Sequential(
            nn.Linear(self._hidden_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        
        self.handcrafted = HandcraftedFeatures(seq_len=23)
        self.feature_combine = nn.Linear(output_dim + self.handcrafted.output_dim, output_dim)
    
    def _apply_lora(self, r, alpha):
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05,
                                      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], bias="none")
            self.backbone = get_peft_model(self.backbone, lora_config)
        except ImportError:
            logger.warning("peft not available, training projection head only")
    
    def _build_fallback(self):
        """Fallback: Small BiGRU encoder."""
        self._hidden_dim = 256
        self.fallback_embedding = nn.Embedding(5, 128)
        self.fallback_pos = nn.Parameter(torch.randn(1, 23, 128) * 0.02)
        self.fallback_gru = nn.GRU(128, self._hidden_dim // 2, num_layers=2,
                                    batch_first=True, bidirectional=True, dropout=0.1)
        self.fallback_norm = nn.LayerNorm(self._hidden_dim)
    
    def forward(self, x: torch.Tensor, sequences: list[str] | None = None) -> torch.Tensor:
        if self._pretrained_loaded and sequences is not None:
            tokens = self.tokenizer(
                sequences, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(next(self.backbone.parameters()).device)
            outputs = self.backbone(**tokens)
            h = outputs.last_hidden_state.mean(dim=1)
        elif hasattr(self, 'fallback_gru'):
            tokens = x.argmax(dim=1)
            emb = self.fallback_embedding(tokens) + self.fallback_pos[:, :tokens.shape[1], :]
            h, _ = self.fallback_gru(emb)
            h = self.fallback_norm(h.mean(dim=1))
        else:
            h = torch.zeros(x.shape[0], self._hidden_dim, device=x.device)
        
        z_backbone = self.projection(h)
        z_hand = self.handcrafted(x)
        z_s = self.feature_combine(torch.cat([z_backbone, z_hand], dim=-1))
        return z_s


# ═══════════════════════════════════════════════════════════════
# Fixed Nucleotide Transformer Encoder v3
# ═══════════════════════════════════════════════════════════════

class NucleotideTransformerEncoderV3(nn.Module):
    """Fixed NT encoder with proper error handling and fallback."""
    
    def __init__(
        self,
        model_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        output_dim: int = 128,
        freeze_layers: int = 20,  # Freeze most of the 500M model
        dropout: float = 0.1,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self._hidden_dim = 1024
        self._pretrained_loaded = False
        
        try:
            from transformers import AutoModel, AutoTokenizer
            logger.info(f"Loading Nucleotide Transformer from {model_name}...")
            kwargs = {"trust_remote_code": True}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            self.backbone = AutoModel.from_pretrained(model_name, **kwargs)
            
            if hasattr(self.backbone.config, 'hidden_size'):
                self._hidden_dim = self.backbone.config.hidden_size
            
            if freeze_layers > 0:
                frozen_count = 0
                for name, param in self.backbone.named_parameters():
                    should_freeze = False
                    if 'embeddings' in name:
                        should_freeze = True
                    for i in range(freeze_layers):
                        if f'layer.{i}.' in name or f'encoder.layer.{i}.' in name:
                            should_freeze = True
                    if should_freeze:
                        param.requires_grad = False
                        frozen_count += 1
                logger.info(f"Frozen {frozen_count} parameters")
            
            self._pretrained_loaded = True
            logger.info(f"NT loaded. Hidden dim: {self._hidden_dim}")
            
        except Exception as e:
            logger.error(f"FAILED to load NT: {e}")
            self.tokenizer = None
            self.backbone = None
            self._build_fallback()
        
        self.projection = nn.Sequential(
            nn.Linear(self._hidden_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        
        self.handcrafted = HandcraftedFeatures(seq_len=23)
        self.feature_combine = nn.Linear(output_dim + self.handcrafted.output_dim, output_dim)
    
    def _build_fallback(self):
        self._hidden_dim = 256
        self.fallback_embedding = nn.Embedding(5, 128)
        self.fallback_pos = nn.Parameter(torch.randn(1, 23, 128) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.fallback_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fallback_proj = nn.Linear(128, self._hidden_dim)
        self.fallback_norm = nn.LayerNorm(self._hidden_dim)
    
    def forward(self, x: torch.Tensor, sequences: list[str] | None = None) -> torch.Tensor:
        if self._pretrained_loaded and sequences is not None:
            tokens = self.tokenizer(
                sequences, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(next(self.backbone.parameters()).device)
            outputs = self.backbone(**tokens)
            h = outputs.last_hidden_state.mean(dim=1)
        elif hasattr(self, 'fallback_transformer'):
            tokens = x.argmax(dim=1)
            emb = self.fallback_embedding(tokens) + self.fallback_pos[:, :tokens.shape[1], :]
            h = self.fallback_transformer(emb)
            h = self.fallback_proj(h.mean(dim=1))
            h = self.fallback_norm(h)
        else:
            h = torch.zeros(x.shape[0], self._hidden_dim, device=x.device)
        
        z_backbone = self.projection(h)
        z_hand = self.handcrafted(x)
        z_s = self.feature_combine(torch.cat([z_backbone, z_hand], dim=-1))
        return z_s


# ═══════════════════════════════════════════════════════════════
# Factory function v3
# ═══════════════════════════════════════════════════════════════

def build_sequence_encoder_v3(cfg) -> nn.Module:
    """Factory function to build v3 sequence encoder."""
    encoder_type = cfg.type.lower()
    output_dim = getattr(cfg, 'output_dim', 128)
    cache_dir = getattr(cfg, 'cache_dir', None)
    
    if encoder_type == "cnn_gru":
        return CNNGRUEncoderV3(
            kernel_sizes=list(getattr(cfg.cnn, 'kernel_sizes', [1, 3, 5, 7])),
            n_filters=getattr(cfg.cnn, 'n_filters', 128),
            gru_hidden=getattr(cfg.gru, 'hidden_size', 256),
            gru_layers=getattr(cfg.gru, 'num_layers', 2),
            output_dim=output_dim,
            cnn_dropout=getattr(cfg.cnn, 'dropout', 0.1),
            gru_dropout=getattr(cfg.gru, 'dropout', 0.2),
        )
    elif encoder_type == "dnabert2":
        return DNABERT2EncoderV3(output_dim=output_dim, cache_dir=cache_dir)
    elif encoder_type == "nucleotide_transformer":
        return NucleotideTransformerEncoderV3(output_dim=output_dim, cache_dir=cache_dir)
    elif encoder_type == "caduceus":
        return CaduceusEncoderV3(output_dim=output_dim)
    elif encoder_type == "evo":
        return EvoEncoderV3(output_dim=output_dim, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown sequence encoder type: {encoder_type}")
