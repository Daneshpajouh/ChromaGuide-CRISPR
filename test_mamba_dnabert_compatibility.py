#!/usr/bin/env python3
"""
Test Mamba-2 Compatibility with DNABERT-2 Embeddings
Validates that BiMamba2 can process 768-dim BERT outputs
"""
import torch
import torch.nn as nn
import sys
import os

# Add project to path
sys.path.insert(0, '/Users/studio/Desktop/PhD/Proposal')

from src.model.bimamba2_block import BiMamba2
from src.model.mamba2_block import Mamba2Config

# ============================================================================
# HYBRID ARCHITECTURE TEST
# ============================================================================

class HybridDNABERTMamba(nn.Module):
    """
    Hybrid architecture: DNABERT-2 (frozen) → Mamba-2 (trainable)

    Architecture:
        Input Sequence → DNABERT-2 (768-dim embeddings) → Mamba-2 Layers → Output
    """

    def __init__(self, dnabert2_model, n_mamba_layers=4):
        super().__init__()

        # DNABERT-2 encoder (can be frozen)
        self.dnabert = dnabert2_model

        # Get BERT hidden size
        bert_dim = self.dnabert.config.hidden_size  # Should be 768

        print(f"DNABERT-2 hidden size: {bert_dim}")

        # Mamba-2 configuration (must match BERT dimension)
        self.mamba_config = Mamba2Config(
            d_model=bert_dim,  # CRITICAL: Must match BERT!
            n_layer=n_mamba_layers,
            d_state=64,  # State space dimension
            expand=2,
            headdim=64,
            chunk_size=256,
            ssm_cfg={
                "dt_min": 0.001,
                "dt_max": 0.1,
            }
        )

        # Stack of BiMamba2 layers
        self.mamba_layers = nn.ModuleList([
            BiMamba2(self.mamba_config)
            for _ in range(n_mamba_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(bert_dim)
            for _ in range(n_mamba_layers)
        ])

        # Output head for CRISPR efficiency
        self.output_head = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, freeze_bert=True):
        """
        Forward pass

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            freeze_bert: Whether to freeze BERT weights

        Returns:
            efficiency: (batch_size, 1)
        """
        # DNABERT-2 encoding
        if freeze_bert:
            with torch.no_grad():
                bert_outputs = self.dnabert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        else:
            bert_outputs = self.dnabert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Get hidden states (batch_size, seq_len, 768)
        hidden_states = bert_outputs.last_hidden_state

        print(f"BERT output shape: {hidden_states.shape}")

        # Pass through Mamba-2 layers
        x = hidden_states
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            # BiMamba2 returns (output, state)
            mamba_out, _ = mamba_layer(x)

            # Residual connection + LayerNorm
            x = layer_norm(x + mamba_out)

            print(f"After Mamba layer: {x.shape}")

        # Pool to (batch_size, 768) - use [CLS] token
        pooled = x[:, 0, :]

        print(f"Pooled shape: {pooled.shape}")

        # Predict efficiency
        output = self.output_head(pooled)

        print(f"Output shape: {output.shape}")

        return output

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_compatibility():
    """Test if Mamba-2 can process DNABERT-2 embeddings"""

    print("="*80)
    print("TESTING DNABERT-2 + MAMBA-2 COMPATIBILITY")
    print("="*80)

    try:
        # Load DNABERT-2
        print("\n1. Loading DNABERT-2...")
        from transformers import AutoModel, AutoTokenizer

        model_name = "zhihan1996/DNABERT-2-117M"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dnabert = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        print(f"✓ DNABERT-2 loaded")
        print(f"   Config: {dnabert.config}")

        # Create hybrid model
        print("\n2. Creating hybrid DNABERT-2 + Mamba-2 model...")
        hybrid_model = HybridDNABERTMamba(dnabert, n_mamba_layers=2)

        print(f"✓ Hybrid model created")

        # Test forward pass
        print("\n3. Testing forward pass...")

        # Create dummy input
        test_sequence = "ATCGATCGATCGTAGCTAGCTAGGGATCG"
        inputs = tokenizer(
            test_sequence,
            return_tensors='pt',
            padding='max_length',
            max_length=64,
            truncation=True
        )

        print(f"   Input IDs shape: {inputs['input_ids'].shape}")
        print(f"   Attention mask shape: {inputs['attention_mask'].shape}")

        # Forward pass
        with torch.no_grad():
            output = hybrid_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                freeze_bert=True
            )

        print(f"\n✓ Forward pass successful!")
        print(f"   Output: {output}")
        print(f"   Shape: {output.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in hybrid_model.parameters())
        trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)

        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters: {total_params - trainable_params:,}")

        print(f"\n{'='*80}")
        print(f"✅ COMPATIBILITY TEST PASSED!")
        print(f"{'='*80}")
        print(f"\nDNABERT-2 (768-dim) → Mamba-2 works correctly!")
        print(f"Ready for hybrid training.")

        return True

    except Exception as e:
        print(f"\n❌ COMPATIBILITY TEST FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# DIMENSION MISMATCH FIX
# ============================================================================

def create_dimension_adapter(input_dim, output_dim):
    """Create adapter layer if dimensions don't match"""
    if input_dim == output_dim:
        return nn.Identity()
    else:
        return nn.Linear(input_dim, output_dim)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Testing Mamba-2 compatibility with DNABERT-2 embeddings...\n")

    success = test_compatibility()

    if success:
        print("\n✅ Next steps:")
        print("1. Train baseline DNABERT-2 model")
        print("2. Compare performance: DNABERT-2 alone vs DNABERT-2 + Mamba-2")
        print("3. If Mamba adds value (>5% improvement), use hybrid")
    else:
        print("\n⚠️  Fix compatibility issues before proceeding")
        print("Check dimension mismatches in src/model/bimamba2_block.py")
