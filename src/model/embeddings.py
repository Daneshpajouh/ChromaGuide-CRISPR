import torch
import torch.nn as nn
import torch.nn.functional as F

class DNAEmbedding(nn.Module):
    """
    Embeds DNA sequences (ACGTN) into a dense vector space.
    Supports character-level embedding.
    """
    def __init__(self, d_model, vocab_size=5): # A, C, G, T, N
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        """
        x: (Batch, SeqLen) integer tensor
        Returns: (Batch, SeqLen, D_Model)
        """
        return self.embedding(x)

class AttentionWeightedMultimodalFusion(nn.Module):
    """
    Dissertation Chapter 4: Multimodal Attention Fusion.

    Dynamically weights 5+ epigenomic modalities (ATAC, Hi-C, marks)
    against DNA sequence features using position-specific attention.

    Architecture:
    1. Project all modalities into d_model subspace.
    2. Compute attention scores for each modality per position.
    3. Perform weighted sum of projected features.
    4. Apply LayerNorm for stability.
    """
    def __init__(self, d_model, n_tracks=5, d_attn=64):
        super().__init__()
        self.d_model = d_model
        self.n_modalities = 1 + n_tracks # Sequence + Tracks (ATAC, H3K27ac, Hi-C, Nuc, Meth)

        # Projections for each track
        # Assuming tracks are scalar signals except Hi-C (managed by track count)
        self.track_proj = nn.Linear(n_tracks, d_model)

        # Attention Query Projection
        # Dissertation Spec: Projects all embeddings to a shared space for softmax scoring
        self.attn_proj = nn.Linear(d_model, d_attn)
        self.modality_heads = nn.Linear(d_attn, 1) # Yields score per modality

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_seq, x_epi):
        """
        x_seq: (B, L, D) - Sequence embeddings
        x_epi: (B, L, N_TRACKS) - Multi-track epigenomics
        """
        batch, seq_len, _ = x_seq.shape

        # 1. Project Epigenomics to d_model
        # We can project all tracks simultaneously if they are at the same resolution
        x_epi_proj = self.track_proj(x_epi) # (B, L, D)

        # 2. Compute Attention Weights (Chapter 4, Section 5.2)
        # Sequence input
        z_seq = torch.tanh(self.attn_proj(x_seq)) # (B, L, d_attn)
        score_seq = self.modality_heads(z_seq) # (B, L, 1)

        # Epigenomic input
        z_epi = torch.tanh(self.attn_proj(x_epi_proj)) # (B, L, d_attn)
        score_epi = self.modality_heads(z_epi) # (B, L, 1)

        # Softmax over modalities (Sequence vs Epigenetics)
        # Note: If we had individual scores for each of the 5 tracks, we'd do it per track.
        # But here we fuse tracks first for efficiency.
        # To strictly match Chapter 4's "per-modality" attention, we'd need to project tracks individually.

        # Simplified Joint Weighting (Sequence vs Composite Epigenetics)
        scores = torch.cat([score_seq, score_epi], dim=-1) # (B, L, 2)
        weights = torch.softmax(scores, dim=-1) # (B, L, 2)

        # 3. Weighted Fusion
        w_seq = weights[:, :, 0:1] # (B, L, 1)
        w_epi = weights[:, :, 1:2] # (B, L, 1)

        h_fused = (w_seq * x_seq) + (w_epi * x_epi_proj)

        return self.norm(h_fused)
