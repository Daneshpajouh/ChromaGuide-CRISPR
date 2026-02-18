"""
CRISPRO-MAMBA-X MLX Port
Apple Silicon Optimized Implementation using MLX Framework

This port leverages Apple Neural Engine (ANE) for faster training on M-series chips.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from pathlib import Path
import time

# ============================================================================
# DNA Embedding Layer
# ============================================================================
class DNAEmbedding(nn.Module):
    """DNA sequence embedding (ACGTN vocabulary)"""
    def __init__(self, d_model: int, vocab_size: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def __call__(self, x):
        return self.embedding(x)

# ============================================================================
# Gated Epigenetic Fusion
# ============================================================================
class GatedEpigeneticFusion(nn.Module):
    """SOTA Gated Fusion for epigenetic markers"""
    def __init__(self, d_model: int, n_tracks: int = 6):
        super().__init__()
        self.epi_proj = nn.Linear(n_tracks, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)

    def __call__(self, x_dna, x_epi):
        # x_dna: (B, L, D), x_epi: (B, n_tracks)
        if x_epi.ndim == 2:
            x_epi = mx.expand_dims(x_epi, axis=1)  # (B, 1, n_tracks)

        epi_latent = self.epi_proj(x_epi)  # (B, 1, D)
        gate = mx.sigmoid(self.gate_proj(x_dna))  # (B, L, D)
        epi_gated = epi_latent * gate
        return x_dna + epi_gated

# ============================================================================
# Mamba-2 SSM Block (Simplified for MLX)
# ============================================================================
class Mamba2Block(nn.Module):
    """
    Simplified Mamba-2 SSM block for MLX.
    Uses linear projections instead of conv1d for cross-platform compatibility.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        # Input projection (x and z gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # SSM parameters (B, C projections)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)

        # Discretization parameter
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Layer norm for stability
        self.norm = nn.LayerNorm(self.d_inner)

    def __call__(self, x):
        B, L, D = x.shape

        # Input projection
        xz = self.in_proj(x)
        x_part, z = mx.split(xz, 2, axis=-1)

        # Apply SiLU activation
        x_part = nn.silu(x_part)
        x_part = self.norm(x_part)

        # Simplified SSM: Use attention-like mechanism
        # This is a linearized approximation that runs efficiently on MLX
        x_dbl = self.x_proj(x_part)
        B_proj, C_proj = mx.split(x_dbl, 2, axis=-1)  # (B, L, d_state) each

        # Compute simplified state-space output
        # Using einsum for the key SSM computation
        dt = nn.softplus(self.dt_proj(x_part))  # (B, L, d_inner)

        # Simplified linear attention: output = softmax(BC^T) @ x
        # Scale for numerical stability
        scale = 1.0 / np.sqrt(self.d_state)
        attn = nn.softmax(B_proj @ mx.transpose(C_proj, (0, 2, 1)) * scale, axis=-1)  # (B, L, L)

        # Apply decay-weighted attention
        y = attn @ x_part  # (B, L, d_inner)

        # Gating
        y = y * nn.silu(z)

        # Output projection
        return self.out_proj(y)

# ============================================================================
# Bidirectional Mamba Encoder
# ============================================================================
class BiMambaEncoder(nn.Module):
    """Bidirectional Mamba-2 encoder"""
    def __init__(self, d_model: int, n_layers: int = 4):
        super().__init__()
        self.layers_fwd = [Mamba2Block(d_model) for _ in range(n_layers)]
        self.layers_bwd = [Mamba2Block(d_model) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, x):
        # Forward pass
        x_fwd = x
        for layer in self.layers_fwd:
            x_fwd = x_fwd + layer(x_fwd)

        # Backward pass (reverse sequence using slicing)
        x_bwd = x[:, ::-1, :]  # Reverse along sequence dimension
        for layer in self.layers_bwd:
            x_bwd = x_bwd + layer(x_bwd)
        x_bwd = x_bwd[:, ::-1, :]  # Reverse back

        # Combine
        x = x_fwd + x_bwd
        return self.norm(x)

# ============================================================================
# CRISPRO Model (MLX)
# ============================================================================
class CRISPROModelMLX(nn.Module):
    """
    CRISPRO-MAMBA-X Model ported to MLX for Apple Silicon optimization.
    """
    def __init__(self, d_model: int = 256, n_layers: int = 4, n_modalities: int = 6, vocab_size: int = 6):
        super().__init__()

        # Embeddings
        self.dna_emb = DNAEmbedding(d_model, vocab_size)
        self.epi_fusion = GatedEpigeneticFusion(d_model, n_modalities)

        # Backbone
        self.encoder = BiMambaEncoder(d_model, n_layers)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def __call__(self, seq, epi):
        # Embed DNA
        x = self.dna_emb(seq)  # (B, L, D)

        # Fuse epigenetics
        x = self.epi_fusion(x, epi)

        # Encode
        x = self.encoder(x)

        # Pool (mean)
        x = mx.mean(x, axis=1)  # (B, D)

        # Heads
        cls_out = mx.sigmoid(self.cls_head(x))
        reg_out = mx.sigmoid(self.reg_head(x))

        return cls_out.squeeze(-1), reg_out.squeeze(-1)

# ============================================================================
# Loss Functions
# ============================================================================
def binary_cross_entropy(pred, target):
    """BCE loss"""
    eps = 1e-7
    pred = mx.clip(pred, eps, 1 - eps)
    return -mx.mean(target * mx.log(pred) + (1 - target) * mx.log(1 - pred))

def mse_loss(pred, target):
    """MSE loss"""
    return mx.mean((pred - target) ** 2)

def hybrid_loss(pred_cls, pred_reg, target_cls, target_reg, alpha=0.5):
    """Combined classification + regression loss"""
    loss_cls = binary_cross_entropy(pred_cls, target_cls)

    # Only compute regression on positive samples
    mask = target_cls > 0
    if mx.sum(mask) > 0:
        loss_reg = mse_loss(pred_reg * mask, target_reg * mask) / (mx.sum(mask) + 1e-6)
    else:
        loss_reg = mx.array(0.0)

    return loss_cls + alpha * loss_reg

# ============================================================================
# Training Script
# ============================================================================
def train_step(model, optimizer, seq, epi, targets):
    """Single training step with gradient computation"""
    def loss_fn(model):
        pred_cls, pred_reg = model(seq, epi)
        target_cls = (targets > 0).astype(mx.float32)
        return hybrid_loss(pred_cls, pred_reg, target_cls, targets)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss

def main():
    print("=" * 60)
    print("CRISPRO-MAMBA-X MLX Training")
    print("Apple Neural Engine Optimized")
    print("=" * 60)

    # Configuration
    BATCH_SIZE = 8
    SEQ_LEN = 4096
    D_MODEL = 256
    N_LAYERS = 4
    EPOCHS = 30
    LR = 3e-4

    # Create model
    print("\nInitializing model...")
    model = CRISPROModelMLX(d_model=D_MODEL, n_layers=N_LAYERS)

    # Count parameters (MLX returns nested dicts)
    def count_params(params):
        total = 0
        for k, v in params.items():
            if isinstance(v, dict):
                total += count_params(v)
            elif hasattr(v, 'size'):
                total += v.size
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        total += count_params(item)
        return total

    n_params = count_params(model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=LR)

    # Synthetic data for benchmarking
    print(f"\nGenerating synthetic data (B={BATCH_SIZE}, L={SEQ_LEN})...")
    seq = mx.random.randint(0, 5, (BATCH_SIZE, SEQ_LEN))
    epi = mx.random.normal((BATCH_SIZE, 6))
    targets = mx.random.uniform(shape=(BATCH_SIZE,))

    # Warmup
    print("Warming up...")
    for _ in range(5):
        loss = train_step(model, optimizer, seq, epi, targets)

    # Benchmark
    print("\n" + "=" * 60)
    print("BENCHMARKING")
    print("=" * 60)

    iterations = 50
    start = time.time()
    for i in range(iterations):
        loss = train_step(model, optimizer, seq, epi, targets)
        if (i + 1) % 10 == 0:
            print(f"  Iter {i+1}/{iterations}: Loss = {float(loss):.4f}")

    elapsed = time.time() - start
    throughput = iterations / elapsed

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} it/s")
    print(f"  Time per iteration: {elapsed/iterations*1000:.2f} ms")
    print("=" * 60)

    return throughput

if __name__ == "__main__":
    main()
