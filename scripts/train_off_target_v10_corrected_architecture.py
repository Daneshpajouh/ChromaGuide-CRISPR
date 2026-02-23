#!/usr/bin/env python3
"""
CORRECTED V10: Off-Target Prediction with REAL CRISPR_DNABERT Architecture
Based on Kimata et al. (2025) PLOS ONE (PMID: 41223195)

CRITICAL CORRECTIONS from placeholder architecture:
1. Epigenetic features: 300-dim (not 690): 3 marks × 100 bins each
   - ATAC-seq: 100 bins (500bp window, 10bp bins)
   - H3K4me3: 100 bins (500bp window, 10bp bins)  
   - H3K27ac: 100 bins (500bp window, 10bp bins)

2. Per-mark architecture: Separate encoder + gate per epigenetic mark
   - Each encoder: Linear(100,256)->ReLU->Drop->Linear(256,512)->...->Linear(512,256)
   - Each gate: Linear(776,256)->ReLU->...->Linear(512,256)->Sigmoid()
   - 776 = DNABERT(768) + mismatch(7) + bulge(1)

3. Classifier: Linear(1536, 2) where 1536 = 768 DNABERT + 256*3 marks

4. Training: batch_size=128, epochs=8, lr=2e-5 DNABERT, lr=1e-3 epi layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PerMarkEpigenicGating(nn.Module):
    """
    Gating mechanism for a SINGLE epigenetic mark (100 dimensions)
    matches the exact architecture from Kimata et al. (2025)
    """
    def __init__(self, mark_dim=100, hidden_dim=256, dnabert_dim=768, dropout=0.1):
        super().__init__()
        
        # Step 1: Encoder for single mark (100 -> 256)
        self.encoder = nn.Sequential(
            nn.Linear(mark_dim, hidden_dim),  # 100 -> 256
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim * 2),  # 256 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),  # 512 -> 1024
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim)  # 512 -> 256
        )
        
        # Step 2: Gate mechanism  
        # Input: DNABERT[CLS](768) + mismatch(7) + bulge(1) = 776
        gate_input_dim = dnabert_dim + 7 + 1  # 776
        
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),  # 776 -> 256
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim * 2),  # 256 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),  # 512 -> 1024
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, 1),  # 512 -> 1
            nn.Sigmoid()
        )
        
        # Initialize gate bias to -3.0 (conservative gating from paper)
        self.gate[-2].bias.data.fill_(-3.0)
    
    def forward(self, dnabert_cls, mark_features, mismatch_features=None, bulge_features=None):
        """
        dnabert_cls: (batch, 768) - [CLS] token from DNABERT
        mark_features: (batch, 100) - one epigenetic mark (100 bins)
        mismatch_features: (batch, 7) - guide-target mismatch one-hot
        bulge_features: (batch, 1) - bulge presence/absence
        
        Returns: (batch, 256) - gated epigenetic features
        """
        # Encode the mark
        encoded = self.encoder(mark_features)  # (batch, 256)
        
        # Create gate input if not provided
        if mismatch_features is None:
            mismatch_features = torch.zeros(dnabert_cls.size(0), 7, 
                                           device=dnabert_cls.device, dtype=dnabert_cls.dtype)
        if bulge_features is None:
            bulge_features = torch.zeros(dnabert_cls.size(0), 1,
                                        device=dnabert_cls.device, dtype=dnabert_cls.dtype)
        
        # Gate input: concatenate DNABERT + mismatch + bulge
        gate_input = torch.cat([dnabert_cls, mismatch_features, bulge_features], dim=1)  # (batch, 776)
        gate_output = self.gate(gate_input)  # (batch, 1)
        
        # Apply gate: weighted interpolation between zero and encoded features
        gated = encoded * gate_output  # (batch, 256)
        
        return gated


class DNABERTOffTargetCorrected(nn.Module):
    """
    CORRECTED V10 Off-Target Classifier - Exact architecture from Kimata et al. (2025)
    
    Architecture:
    1. DNABERT-2 -> [CLS] token (768 dims)
    2. THREE separate epigenetic gating modules (1 per mark):
       - ATAC-seq (100 dims per mark) -> encoded to 256
       - H3K4me3 (100 dims per mark) -> encoded to 256
       - H3K27ac (100 dims per mark) -> encoded to 256
    3. Classifier: Linear(768 + 256*3, 2) = Linear(1536, 2)
    """
    
    def __init__(self, dnabert_model_name="zhihan1996/DNABERT-2-117M", 
                 use_cnn_bilstm=False, hidden_dim=256, dropout=0.1):
        """
        dnabert_model_name: HuggingFace model ID
        use_cnn_bilstm: If True, add CNN+BiLSTM (not in original Kimata paper)
        hidden_dim: Hidden dimension for encoders (256 per Kimata et al.)
        dropout: Dropout rate
        """
        super().__init__()
        
        # Import here to avoid circular deps
        from transformers import AutoTokenizer, AutoModel
        import os
        
        # Load DNABERT-2
        local_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dnabert2')
        if os.path.exists(local_model_path):
            print(f"Loading DNABERT-2 from local cache: {local_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.dnabert = AutoModel.from_pretrained(local_model_path)
        else:
            print(f"Local model not found at {local_model_path}, using HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(dnabert_model_name, trust_remote_code=True)
            self.dnabert = AutoModel.from_pretrained(dnabert_model_name, trust_remote_code=True)
        
        self.dnabert_dim = self.dnabert.config.hidden_size  # 768
        
        # Freeze DNABERT, unfreeze last 6 layers for fine-tuning
        for param in self.dnabert.parameters():
            param.requires_grad = False
        for param in self.dnabert.encoder.layer[-6:].parameters():
            param.requires_grad = True
        
        # THREE separate epigenetic gating modules (one per mark)
        # Each handles 100-dim input from a single epigenetic mark
        self.epi_gating = nn.ModuleDict({
            'atac': PerMarkEpigenicGating(mark_dim=100, hidden_dim=hidden_dim, 
                                         dnabert_dim=self.dnabert_dim, dropout=dropout),
            'h3k4me3': PerMarkEpigenicGating(mark_dim=100, hidden_dim=hidden_dim,
                                            dnabert_dim=self.dnabert_dim, dropout=dropout),
            'h3k27ac': PerMarkEpigenicGating(mark_dim=100, hidden_dim=hidden_dim,
                                            dnabert_dim=self.dnabert_dim, dropout=dropout)
        })
        
        # Classifier: taking DNABERT[CLS] + 3 gated epi marks
        # Input: 768 (DNABERT) + 256*3 (3 marks) = 1536
        self.classifier = nn.Linear(self.dnabert_dim + hidden_dim * 3, 2)
    
    def forward(self, sequences, epi_features, mismatch_features=None, bulge_features=None):
        """
        sequences: List of DNA sequences (will be tokenized)
        epi_features: (batch, 300) - concatenated [atac_100 | h3k4me3_100 | h3k27ac_100]
        mismatch_features: (batch, 7) optional
        bulge_features: (batch, 1) optional
        
        Returns: logits (batch, 2)
        """
        # Tokenize and encode with DNABERT
        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True,
                               truncation=True, max_length=24)
        
        # Move tokens to same device as model
        tokens = {k: v.to(self.dnabert.device) for k, v in tokens.items()}
        
        dnabert_output = self.dnabert(**tokens)
        dnabert_cls = dnabert_output.last_hidden_state[:, 0, :]  # (batch, 768)
        
        # Split 300-dim epi_features into 3 marks (100 dims each)
        atac_feats = epi_features[:, 0:100]        # (batch, 100)
        h3k4me3_feats = epi_features[:, 100:200]   # (batch, 100)
        h3k27ac_feats = epi_features[:, 200:300]   # (batch, 100)
        
        # Process each epigenetic mark through its own gating module
        gated_atac = self.epi_gating['atac'](dnabert_cls, atac_feats, 
                                             mismatch_features, bulge_features)
        gated_h3k4me3 = self.epi_gating['h3k4me3'](dnabert_cls, h3k4me3_feats,
                                                   mismatch_features, bulge_features)
        gated_h3k27ac = self.epi_gating['h3k27ac'](dnabert_cls, h3k27ac_feats,
                                                   mismatch_features, bulge_features)
        
        # Concatenate DNABERT[CLS] + all 3 gated marks
        combined = torch.cat([dnabert_cls, gated_atac, gated_h3k4me3, gated_h3k27ac], dim=1)
        # Shape: (batch, 768 + 256*3 = 1536)
        
        # Classification
        logits = self.classifier(combined)  # (batch, 2)
        
        return logits


if __name__ == "__main__":
    print("CORRECTED V10 Architecture Module")
    print("=" * 70)
    print("\nKey Corrections:")
    print("✓ Epigenetic features: 300-dim (3 marks × 100 bins)")
    print("✓ Per-mark gating: Separate encoder + gate per epigenetic mark")
    print("✓ Classifier input: 1536 dims (768 DNABERT + 256*3)")
    print("✓ Training: batch_size=128, epochs=8, lr=2e-5 DNABERT, lr=1e-3 epi")
    print("✓ Data format: [atac_100 | h3k4me3_100 | h3k27ac_100]")
