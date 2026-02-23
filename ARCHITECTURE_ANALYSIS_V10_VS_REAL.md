# CRITICAL ARCHITECTURE ANALYSIS: V10 vs VERIFIED CRISPR_DNABERT

## Current V10 Implementation Issues

### 1. EPIGENETIC FEATURE DIMENSION - WRONG
**Current Code (Line 209):**
```python
epi_feature_dim=690  # PLACEHOLDER
epi = np.random.randn(690).astype(np.float32)  # Random noise!
```

**REAL Specification (Kimata et al. 2025):**
- Epigenetic marks: ATAC-seq, H3K4me3, H3K27ac (3 marks)
- Window size: 500bp, bin size: 10bp
- Per mark dimension: 500 * 2 / 10 = 100 bins
- **Total epigenetic feature dimension: 3 × 100 = 300 dims**

### 2. EPIGENETIC MODULE ARCHITECTURE - FUNDAMENTALLY WRONG
**Current Code (Lines 85-147):**
```python
class EpigenoticGatingModule:
    # Single nn.Linear encoder treating all 690 dims as one input
    self.epi_encoder = nn.Sequential(
        nn.Linear(feature_dim, epi_hidden_dim),  # Linear(690, 256)
        ...
    )
    # One gate for the whole thing
    self.gating_module = nn.Sequential(
        nn.Linear(gate_input_dim, epi_hidden_dim),  # Linear(776, 256)
        ...
    )
```

**REAL Architecture (Kimata et al. 2025):**
```python
# nn.ModuleDict with SEPARATE encoders per epigenetic mark
self.epi_encoders = nn.ModuleDict({
    'atac': nn.Sequential(Linear(100, 256)->ReLU->Drop->...->Linear(512, 256)),
    'h3k4me3': nn.Sequential(Linear(100, 256)->ReLU->Drop->...->Linear(512, 256)),
    'h3k27ac': nn.Sequential(Linear(100, 256)->ReLU->Drop->...->Linear(512, 256))
})

# Separate gate per mark
self.gating_layers = nn.ModuleDict({
    'atac': nn.Sequential(Linear(776, 256)->ReLU->Drop->...->Sigmoid()),
    'h3k4me3': nn.Sequential(Linear(776, 256)->ReLU->Drop->...->Sigmoid()),
    'h3k27ac': nn.Sequential(Linear(776, 256)->ReLU->Drop->...->Sigmoid())
})

# Initialize gate bias to -3.0 per mark
for mark in self.gating_layers:
    self.gating_layers[mark][-2].bias.data.fill_(-3.0)
```

### 3. DATA PIPELINE - USING PLACEHOLDERS
**Current Code (Line 356):**
```python
# For now, use placeholder epigenetic features
epi = np.random.randn(690).astype(np.float32)  # WRONG: 690 random dimensions
```

**REAL Implementation:**
Should load actual epigenetic features from GSM files:
- GSM4498611 (ATAC-seq): 100 dimensions
- GSM4495703 (H3K4me3): 100 dimensions
- GSM4495711 (H3K27ac): 100 dimensions
- Total: 300 dimensions (properly structured with 3 separate vectors)

### 4. CLASSIFIER ARCHITECTURE - WRONG INPUT DIM
**Current Code (Line 278-281):**
```python
self.classifier = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(hidden_dim * 4, 2)  # Input: 1024 (256*4)
)
```

**REAL Architecture:**
```python
# Classifier input: DNABERT[CLS] + 3 gated epi marks
# = 768 + 256*3 = 1536 dimensions
self.classifier = nn.Linear(768 + 256*3, 2)
```

### 5. FORWARD PASS - MISSING PER-MARK PROCESSING
**Current Code (Lines 329-335):**
```python
gated_features = self.epi_gating(
    dnabert_out[:, 0, :],  # 768-dim
    epi_features,  # 690-dim (WRONG)
    mismatch_features,
    bulge_features
)  # (batch, 256)

combined = torch.cat([seq_repr, cnn_repr, bilstm_repr, gated_features], dim=1)
```

**REAL Forward Pass:**
```python
# Extract 3 mark vectors from epi_features (300-dim total)
atac_feats = epi_features[:, 0:100]      # 100
h3k4me3_feats = epi_features[:, 100:200] # 100
h3k27ac_feats = epi_features[:, 200:300] # 100

# Process each mark separately
gated_atac = self.epi_gating['atac'](dnabert_out[:, 0, :], atac_feats, ...)
gated_h3k4me3 = self.epi_gating['h3k4me3'](dnabert_out[:, 0, :], h3k4me3_feats, ...)
gated_h3k27ac = self.epi_gating['h3k27ac'](dnabert_out[:, 0, :], h3k27ac_feats, ...)

# Concatenate DNABERT + all 3 gated epi marks
combined = torch.cat([dnabert_out[:, 0, :], gated_atac, gated_h3k4me3, gated_h3k27ac], dim=1)
# Shape: (batch, 768 + 256*3 = 1536)
```

### 6. TRAINING PARAMETERS - PARTIALLY WRONG
**Current Code (Lines 413-415):**
```python
total_steps = epochs * len(train_labels) // 128  # 128 batch size ✓
warmup_steps = max(1, total_steps // 10)
scheduler = CosineAnnealingWarmRestarts(optimizer, ...)
```

**Issues:**
- batch_size=128 ✓ CORRECT
- epochs=50 ✗ WRONG (should be 8 per paper)
- lr=2e-5 for DNABERT ✓ CORRECT
- But no separate lr=1e-3 for epi encoders (need param groups)
- majority_rate=0.2 ✓ CORRECT

---

## SUMMARY OF REQUIRED FIXES

| Issue | Current | Required | Priority |
|-------|---------|----------|----------|
| epi_feature_dim | 690 (random) | 300 (3 marks × 100 bins) | **CRITICAL** |
| Module structure | Single monolithic | nn.ModuleDict per mark | **CRITICAL** |
| Data loading | Random placeholder | Load 3 marks (100 each) | **CRITICAL** |
| Classifier input | 1024 | 1536 (768 + 256*3) | **CRITICAL** |
| Per-mark processing | Missing | Separate gate per mark | **CRITICAL** |
| Epochs | 50 | 8 | Important |
| Optimizer lr groups | Only DNABERT+task | Per-mark separate lr | Important |

---

## IMPACT ON TRAINING

**Current Training Status:** 
- Running with WRONG architecture (epi_feature_dim=690 random noise)
- Will NOT produce meaningful results
- Classifier training on concatenated features of:
  - DNABERT[CLS] (768)
  - CNN (256) 
  - BiLSTM (256)
  - Gated random noise processed through MLP (256)
- This is NOT the REAL CRISPR_DNABERT architecture

**Recommendation:**
- **STOP current training immediately** 
- Apply architecture corrections
- Retrain from scratch with correct 300-dim epigenetic features
- Results will be publication-ready only after this fix

