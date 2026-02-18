import torch
import torch.nn as nn
from src.model.mamba2_block import Mamba2Config
from src.model.bimamba2_block import BiMamba2
from src.model.embeddings import DNAEmbedding, AttentionWeightedMultimodalFusion

class CRISPRO_MambaX(nn.Module):
    """
    CRISPRO-Mamba-X (Dissertation Chapter 6 Apex Model).

    Architectural Specifications:
    - Task: Strategic Genome Editing Prediction (On/Off Target).
    - Context Window: 1.2 Mbp (Megabase-scale chromatin).
    - Backbone: Stack of Bi-Bidirectional Mamba-2 (SSD) blocks.
    - Complexity: O(L * d) linear-time sequence processing.
    - Fusion: Gated Multimodal Epigenomics (ATAC, Hi-C, marks).
    - Output: Conformal-ready point predictions with dual heads.
    """
    def __init__(self,
                 d_model=512,
                 n_layers=6,
                 d_state=128,
                 n_modalities=5,
                 vocab_size=5,
                 chunk_size=256):
        super().__init__()

        self.d_model = d_model

        # 1. Megabase-Scale DNA Embedding
        self.dna_emb = DNAEmbedding(d_model, vocab_size)

        # 2. Gated Multimodal Epigenomics Fusion (Chapter 4)
        # Integrates ATAC, H3K27ac, Hi-C, Nucleosomes, Methylation
        # Upgraded to Attention-Weighted Fusion for SOTA outperformance
        self.epi_fusion = AttentionWeightedMultimodalFusion(d_model, n_tracks=n_modalities)

        # 3. Dynamic Mamba-2 Backbone (Chapter 6)
        # Using SSD (Structured State Space Duality) for linear memory scaling.
        self.config = Mamba2Config(
            d_model=d_model,
            n_layer=n_layers,
            d_state=d_state,
            expand=2,
            headdim=64,
            chunk_size=chunk_size,
            ssm_cfg={
                "dt_min": 0.001,
                "dt_max": 0.1,
                "A_init_range": (1, 16),
                "dt_init_floor": 1e-4,
            }
        )

        self.encoder_layers = nn.ModuleList([
            BiMamba2(self.config) for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # 4. Multi-Head Strategic Output (On-Target & Off-Target)
        # Chapter 7 integration: Joint prediction
        self.on_target_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self.off_target_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        # 5. Global Adaptive Pool for context synthesis
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, dna_seq, epigenetics=None):
        """
        dna_seq: (Batch, SeqLen) - LongContext DNA
        epigenetics: (Batch, SeqLen, n_tracks) - High-dim chromatin tracks
        """
        # 1. Embed Sequence
        x = self.dna_emb(dna_seq) # (B, L, D)

        # 2. Fuse Epigenomics (Gated Fusion)
        if epigenetics is not None:
            x = self.epi_fusion(x, epigenetics) # (B, L, D)

        # 3. Process through Mamba-2 Stack (Linear Complexity)
        for layer in self.encoder_layers:
            x, _ = layer(x)

        x = self.final_norm(x)

        # 4. Context Synthesis (Global Pooling)
        # Megabase context is reduced to strategic latent vector
        x_trans = x.transpose(1, 2) # (B, D, L)
        x_latent = self.pool(x_trans).squeeze(-1) # (B, D)

        # 5. Strategic Predictions
        on_target = self.on_target_head(x_latent) # Efficiency (0-1)
        off_target = self.off_target_head(x_latent) # Risk Probability (0-1)

        return {
            'on_target': torch.sigmoid(on_target),
            'off_target': torch.sigmoid(off_target),
            'latent': x_latent
        }

if __name__ == "__main__":
    # Rapid Unit Test for 1.2 Mbp Context
    print("🧪 Testing Mamba-X Architecture (Megabase Scale)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1.2 Mbp test (Batch=1 for memory safety)
    # Must be multiple of chunk_size (256)
    chunk_size = 256
    L = 1200128 # (256 * 4688)
    model = CRISPRO_MambaX(d_model=256, n_layers=2, chunk_size=chunk_size).to(device)

    dna_mock = torch.randint(0, 5, (1, L)).to(device)
    epi_mock = torch.randn(1, L, 5).to(device)

    print(f"[*] Input Sequence Length: {L}")
    print(f"[*] Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    with torch.no_grad():
        out = model(dna_mock, epi_mock)

    print(f"[*] Success! On-Target: {out['on_target'].item():.4f}")
    print(f"[*] Success! Off-Target: {out['off_target'].item():.4f}")
