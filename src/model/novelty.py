import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.mamba2_block import Mamba2, Mamba2Config
from src.model.embeddings import DNAEmbedding, GatedEpigeneticFusion
import mlx.core as mx # Not used here, this is PyTorch

class MambaMoEBlock(nn.Module):
    """
    Novel Contribution: Mixture-of-Mambas (CRISPRO-MoM)
    Uses a sparse router to select specific Mamba Experts for different sequence contexts.
    Hypothesis: Biological mechanisms (bulges vs mismatches) require different state dynamics.
    """
    def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Config for Mamba2
        cfg = Mamba2Config(d_model=d_model, n_layer=1)

        # Experts: Collection of Mamba-2 blocks
        # Variation: Initialize with slightly different dt_min/max to encourage diversity?
        self.experts = nn.ModuleList([
            Mamba2(cfg) for _ in range(num_experts)
        ])

        # Router: Linear layer to project input to expert logits
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        # x: [Batch, Seq, Model]
        B, T, D = x.shape

        # Compute Router Logits: [B, T, NumExperts]
        router_logits = self.router(x)

        # Softmax for routing weights
        routing_weights = F.softmax(router_logits, dim=-1)

        # Top-k Selection
        # weights: [B, T, k], indices: [B, T, k]
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Execution Strategy: "Soft Mixture" - Run all experts, weight outputs.
        # This preserves the recurrence state for each expert.

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x)) # [B, T, D]

        # Stack: [B, T, D, NumExperts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)

        # Weight outputs: Sum(Weight * ExpertOutput)
        # routing_weights: [B, T, NumExperts]
        # output: [B, T, D]

        # Broadcast weights for multiplication
        # [B, T, 1, NumExperts] * [B, T, D, NumExperts]
        weighted_out = expert_outputs * routing_weights.unsqueeze(2)
        final_output = torch.sum(weighted_out, dim=-1)

        return final_output

class BiMambaMoEEncoder(nn.Module):
    def __init__(self, d_model, n_layers=4, n_experts=4):
        super().__init__()
        self.layers_fwd = nn.ModuleList([MambaMoEBlock(d_model, n_experts) for _ in range(n_layers)])
        self.layers_bwd = nn.ModuleList([MambaMoEBlock(d_model, n_experts) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_fwd = x
        for layer in self.layers_fwd:
            x_fwd = layer(x_fwd) + x_fwd

        x_bwd = torch.flip(x, [1])
        for layer in self.layers_bwd:
            x_bwd = layer(x_bwd) + x_bwd
        x_bwd = torch.flip(x_bwd, [1])

        return self.norm(x_fwd + x_bwd)

class CRISPRO_MoM(nn.Module):
    """
    CRISPRO Mixture-of-Mambas Model (Full Implementation)
    """
    def __init__(self, d_model=256, n_layers=4, n_modalities=6, vocab_size=5, n_experts=4):
        super().__init__()
        self.dna_emb = DNAEmbedding(d_model, vocab_size)
        self.epi_fusion = GatedEpigeneticFusion(d_model, n_tracks=n_modalities)
        self.encoder = BiMambaMoEEncoder(d_model, n_layers, n_experts)

        # Heads (Same as baseline)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, seq, epi_tracks):
        x = self.dna_emb(seq)
        x = self.epi_fusion(x, epi_tracks)
        x = self.encoder(x)

        # Global Pooling
        x_pool = torch.mean(x, dim=1)

        pred_cls = torch.sigmoid(self.cls_head(x_pool)).squeeze(-1)
        pred_reg = torch.sigmoid(self.reg_head(x_pool)).squeeze(-1)

        return pred_cls, pred_reg
