import torch
import torch.nn as nn
from src.model.embeddings import DNAEmbedding, GatedEpigeneticFusion
from src.model.mamba2_block import Mamba2Config
from src.model.bimamba2_block import BiMamba2
from src.model.causal import CausalDecoder
from src.model.quantum import QuantumTunnelingCorrection
from src.model.topological import TopologicalRegularizer

class MambaEncoder(nn.Module):
    """
    Stack of Bi-Directional Mamba-2 (SSD) blocks.
    SOTA Architecture (Caduceus-style).
    """
    def __init__(self, d_model, n_layers=4, d_state=128):
        super().__init__()

        # Mamba-2 Configuration
        # n_modalities is handled by fusion, vocab by embedding.
        # We just need internal dims here.
        self.config = Mamba2Config(
            d_model=d_model,
            n_layer=n_layers,
            d_state=d_state,
            expand=2,
            headdim=64,
            # H100 Stability Fixes
            chunk_size=256,              # Must be power of 2 for Triton kernels
            ssm_cfg={
                "dt_min": 0.001,         # Prevent zero timescale -> Inf in kernel
                "dt_max": 0.1,           # Limit max timescale
                "A_init_range": (1.0, 16.0), # Safer A range
                "dt_init_floor": 1e-4,   # Hard floor for parameters
            }
        )

        self.layers = nn.ModuleList([
            BiMamba2(self.config)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Mamba2 returns (output, cache_state)
        # We carry no cache in training (h=None)
        # We discard cache in output
        for layer in self.layers:
            x, _ = layer(x) # x is (B, L, D)
        return self.norm(x)

class CRISPROModel(nn.Module):
    """
    Unified State Space Model for CRISPR Efficiency Prediction (SOTA-Max).
    Upgrades:
    - Gated Epigenetic Fusion (DNABERT-epi style)
    - Two-Head Output:
        - Class Head (Sigmoid): Probability of Activity (Active/Inactive)
        - Reg Head (ReLU): Predicted Efficiency (0-100)
    """
    def __init__(self, d_model=256, n_layers=4, n_modalities=33, vocab_size=20,
                 use_causal=False, use_quantum=False, use_topo=False):
        super().__init__()

        self.d_model = d_model
        self.use_epi = True
        self.use_quantum = use_quantum

        # 1. Embeddings
        # 1. Embeddings
        self.dna_emb = DNAEmbedding(d_model, vocab_size)

        # 2. Epigenetic Fusion
        self.epi_fusion = GatedEpigeneticFusion(d_model, n_tracks=n_modalities)

        # 3. Backbone (Bi-Mamba-2)
        self.encoder = MambaEncoder(d_model, n_layers=n_layers)

        # 4. Two-Head Architecture
        self.class_head = nn.Linear(d_model, 1) # Renamed to match train.py expects (class_head/reg_head)
        self.reg_head = nn.Linear(d_model, 1)

        # Pools
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Biophysical Gating
        self.physics_proj = nn.Linear(2, d_model)

        # Causal Decoder
        if use_causal:
            self.causal_decoder = CausalDecoder(d_model)

        # Quantum Correction
        if use_quantum:
            self.quantum_layer = QuantumTunnelingCorrection(d_model)

        # Topological Regularization (Loss Function Module)
        if use_topo:
            self.topo_reg = TopologicalRegularizer(lam=0.1)

    # 5. Initialization
        nn.init.constant_(self.reg_head.bias, 0.0)
        nn.init.xavier_uniform_(self.reg_head.weight, gain=0.1)


    def forward(self, x, epigenetics=None, biophysics=None, causal=False, return_latents=False, causal_intervention=None):
        # 1. Embeddings
        x = self.dna_emb(x) # (B, L, D)

        # 2. Fusion
        if self.use_epi and epigenetics is not None:
             x = self.epi_fusion(x, epigenetics)

        # 3. Backbone
        x = self.encoder(x) # (B, L, D)

        # 4. Global Pooling
        # Mamba outputs (B, L, D). We want (B, D).
        # AvgPool1d expects (B, D, L).
        x = x.transpose(1, 2) # (B, D, L)
        x_pooled = self.pool(x).squeeze(-1) # (B, D)

        # 5. Biophysical Gating (Authenticity Pillar)
        if biophysics is not None and hasattr(self, 'physics_proj'):
            # biophysics: (B, 2) -> (B, D)
            gate = torch.sigmoid(self.physics_proj(biophysics))
            x_pooled = x_pooled * gate

        outputs = {'classification': self.class_head(x_pooled), 'regression': self.reg_head(x_pooled)}

        if return_latents:
            outputs['latent'] = x_pooled

        # 8. Causal Decoding (Scientific Rigor Pillar)
        if hasattr(self, 'causal_decoder') and causal:
             # CausalDecoder returns: logits, (h_S, h_C, h_P, h_A, h_R)
             causal_out = self.causal_decoder(x_pooled, intervention=causal_intervention)
             outputs['causal'] = causal_out

        return outputs
