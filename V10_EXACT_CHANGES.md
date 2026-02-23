# EXACT CODE CHANGES NEEDED FOR V10 CORRECTION

## File: train_off_target_v10.py

### CHANGE 1: Replace EpigenoticGatingModule class (Lines 85-174)

**REMOVE (WRONG):**
```python
class EpigenoticGatingModule(nn.Module):
    """Epigenetic Feature Gating Module from CRISPR_DNABERT"""
    def __init__(self, feature_dim, epi_hidden_dim=256, dnabert_hidden_size=768,
                 mismatch_dim=7, bulge_dim=1, dropout=0.1):
        super().__init__()

        self.epi_hidden_dim = epi_hidden_dim

        # Single encoder for all 690 dimensions
        self.epi_encoder = nn.Sequential(
            nn.Linear(feature_dim, epi_hidden_dim),  # 690 -> 256 WRONG!
            ...
        )

        # Single gate
        self.gating_module = nn.Sequential(
            nn.Linear(gate_input_dim, epi_hidden_dim),
            ...
        )
```

**REPLACE WITH (CORRECT):**
```python
class PerMarkEpigenicGating(nn.Module):
    """
    Gating mechanism for a SINGLE epigenetic mark (100 dimensions)
    Matches Kimata et al. (2025) CRISPR_DNABERT architecture
    """
    def __init__(self, mark_dim=100, hidden_dim=256, dnabert_dim=768, dropout=0.1):
        super().__init__()
        
        # Encoder for single mark: 100 -> 256
        self.encoder = nn.Sequential(
            nn.Linear(mark_dim, hidden_dim),                    # 100 -> 256
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim * 2),              # 256 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),          # 512 -> 1024
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),          # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim)               # 512 -> 256
        )
        
        # Gate mechanism: input is DNABERT(768) + mismatch(7) + bulge(1) = 776
        gate_input_dim = dnabert_dim + 7 + 1
        
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),              # 776 -> 256
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim * 2),              # 256 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),          # 512 -> 1024
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),          # 1024 -> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, 1),                       # 512 -> 1
            nn.Sigmoid()
        )
        
        # Initialize gate bias to -3.0 (conservative gating)
        self.gate[-2].bias.data.fill_(-3.0)
    
    def forward(self, dnabert_cls, mark_features, mismatch_features=None, bulge_features=None):
        """
        dnabert_cls: (batch, 768)
        mark_features: (batch, 100) - single epigenetic mark
        mismatch_features: (batch, 7) optional
        bulge_features: (batch, 1) optional
        
        Returns: (batch, 256)
        """
        encoded = self.encoder(mark_features)
        
        if mismatch_features is None:
            mismatch_features = torch.zeros(dnabert_cls.size(0), 7,
                                           device=dnabert_cls.device, dtype=dnabert_cls.dtype)
        if bulge_features is None:
            bulge_features = torch.zeros(dnabert_cls.size(0), 1,
                                        device=dnabert_cls.device, dtype=dnabert_cls.dtype)
        
        gate_input = torch.cat([dnabert_cls, mismatch_features, bulge_features], dim=1)
        gate_output = self.gate(gate_input)
        
        gated = encoded * gate_output
        return gated
```

---

### CHANGE 2: Update model initialization (Lines 209-280)

**CHANGE:**
```python
# OLD (Line 209):
epi_feature_dim=690  # WRONG

# NEW:
epi_feature_dim=300  # CORRECT: 3 marks × 100 bins each
```

**OLD (Lines 268-276) - Remove these:**
```python
        # 4. Epigenetic gating (CRISPR_DNABERT style)
        # Input: DNABERT (768) + mismatch (7) + bulge (1) + epigenomics
        self.epi_gating = EpigenoticGatingModule(
            feature_dim=epi_feature_dim,
            epi_hidden_dim=256,
            dnabert_hidden_size=self.dnabert_dim,
            mismatch_dim=7,
            bulge_dim=1,
            dropout=dropout
        )
```

**NEW (Lines 268-276) - Replace with:**
```python
        # 4. Epigenetic gating - THREE separate modules (one per mark)
        # Each module handles 100-dim input from one epigenetic mark
        self.epi_gating = nn.ModuleDict({
            'atac': PerMarkEpigenicGating(mark_dim=100, hidden_dim=256,
                                         dnabert_dim=self.dnabert_dim, dropout=dropout),
            'h3k4me3': PerMarkEpigenicGating(mark_dim=100, hidden_dim=256,
                                            dnabert_dim=self.dnabert_dim, dropout=dropout),
            'h3k27ac': PerMarkEpigenicGating(mark_dim=100, hidden_dim=256,
                                            dnabert_dim=self.dnabert_dim, dropout=dropout)
        })
```

---

### CHANGE 3: Update classifier (Lines 278-281)  

**OLD:**
```python
        # 5. Classification head (EXACT from CRISPR_DNABERT source)
        # Simplified to Dropout + Linear as in original paper
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, 2)  # Input: 256*4 = 1024
        )
```

**NEW:**
```python
        # 5. Classification head
        # Input: DNABERT[CLS](768) + 3 gated marks(256 each) = 1536
        self.classifier = nn.Linear(self.dnabert_dim + 256 * 3, 2)
```

---

### CHANGE 4: Update forward method (Lines 289-336)

**OLD (Wrong):**
```python
    def forward(self, sequences, epi_features, mismatch_features=None, bulge_features=None):
        ...
        dnabert_out = self.dnabert(**tokens).last_hidden_state  # (batch, seq_len, 768)

        # 1. Project DNABERT [CLS] to hidden_dim
        seq_repr = self.seq_proj(dnabert_out[:, 0, :])  # (batch, 256)

        # 2. Multi-scale CNN features  
        cnn_feat = self.cnn_module(dnabert_out.transpose(1, 2))
        cnn_repr = self.cnn_proj(cnn_feat)  # (batch, 256)

        # 3. BiLSTM context
        bilstm_final, bilstm_seq = self.bilstm(dnabert_out)
        bilstm_repr = self.bilstm_proj(bilstm_final)  # (batch, 256)

        # 4. Epigenetic gating with single module
        gated_features = self.epi_gating(
            dnabert_out[:, 0, :],
            epi_features,  # 690-dim WRONG
            mismatch_features,
            bulge_features
        )  # (batch, 256)

        # 5. Concatenate all features
        combined = torch.cat([seq_repr, cnn_repr, bilstm_repr, gated_features], dim=1)
        # (batch, 1024) WRONG INPUT

        # 6. Classification
        logits = self.classifier(combined)  # (batch, 2)

        return logits
```

**NEW (Correct):**
```python
    def forward(self, sequences, epi_features, mismatch_features=None, bulge_features=None):
        ...
        dnabert_out = self.dnabert(**tokens).last_hidden_state
        dnabert_cls = dnabert_out[:, 0, :]  # (batch, 768)
        
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
        # (batch, 768 + 256*3 = 1536) CORRECT INPUT
        
        # Classification
        logits = self.classifier(combined)  # (batch, 2)
        
        return logits
```

---

### CHANGE 5: Update data loading (Line 356)

**OLD:**
```python
                # Try to extract epigenetic features if available in file
                # For now, use placeholder epigenetic features
                epi = np.random.randn(690).astype(np.float32)  # WRONG
                epis.append(epi)
```

**NEW:**
```python
                # Epigenetic features: 300-dim (3 marks × 100 bins each)
                # [ATAC-seq(100) | H3K4me3(100) | H3K27ac(100)]
                # Initialize as zeros - real data would go here
                epi = np.zeros(300, dtype=np.float32)
                # TODO: Load actual values from epigenomic data if available
                epis.append(epi)
```

---

### CHANGE 6: Update training parameters (Lines 376-376)

**OLD:**
```python
def train_model(model, train_seqs, train_epis, train_labels,
                val_seqs, val_epis, val_labels, epochs=8, seed=0):
```

Keep this signature (epochs=8 is correct).

**OLD (Lines 396-403) - Remove CNN/BiLSTM from optimizer:**
```python
    optimizer = optim.AdamW([
        {'params': model.dnabert.parameters(), 'lr': 2e-5},      # DNABERT
        {'params': model.seq_proj.parameters(), 'lr': 1e-3},     # NOT IN PAPER
        {'params': model.cnn_module.parameters(), 'lr': 1e-3},   # NOT IN PAPER
        {'params': model.bilstm.parameters(), 'lr': 1e-3},       # NOT IN PAPER
        {'params': model.epi_gating.parameters(), 'lr': 1e-3},   # Need per-mark
        {'params': model.classifier.parameters(), 'lr': 2e-5}    # WRONG LR
    ], weight_decay=1e-4)
```

**NEW:**
```python
    optimizer = optim.AdamW([
        {'params': model.dnabert.parameters(), 'lr': 2e-5},      # DNABERT: 2e-5
        {'params': model.epi_gating.parameters(), 'lr': 1e-3},   # EPI MARKS: 1e-3
        {'params': model.classifier.parameters(), 'lr': 1e-3}    # CLASSIFIER: 1e-3
    ], weight_decay=1e-4)
```

---

### CHANGE 7: Update main function (Lines 520-525)

**OLD:**
```python
        model = DNABERTOffTargetV10()
        trained_model, val_auc = train_model(
            model, X_train_seqs, X_train_epis, y_train,
            X_val_seqs, X_val_epis, y_val,
            epochs=50, seed=i  # Extended from paper's 8 epochs
        )
```

**NEW:**
```python
        model = DNABERTOffTargetV10()  # Now uses 300-dim, per-mark gating
        trained_model, val_auc = train_model(
            model, X_train_seqs, X_train_epis, y_train,
            X_val_seqs, X_val_epis, y_val,
            epochs=8, seed=i  # Exact from Kimata et al. (2025)
        )
```

---

## Summary of Changes

| Item | Old Value | New Value | File Location |
|------|-----------|-----------|----------------|
| epi_feature_dim | 690 | 300 | Line 209 |
| epi_encoder | Single Linear(690,256) | Per-mark Linear(100,256) | Class definition |
| epi_gating type | EpigenoticGatingModule | nn.ModuleDict (3 items) | Lines 268-276 |
| num_marks | 1 | 3 (atac, h3k4me3, h3k27ac) | Lines 268-276 |
| classifier input | 1024 | 1536 | Line 281 |
| forward() | Concatenates CNN+BiLSTM+epi | Concatenates only DNABERT+3epi | Lines 289-336 |
| epi data | 690 random | 300 zeros | Line 356 |
| optimizer lr groups | 5 groups (wrong) | 3 groups (correct) | Lines 396-403 |
| epochs | 50 | 8 | Line 529 |

**Total lines affected:** ~60 lines (out of 581 in file)

