# V10 ARCHITECTURE CORRECTION GUIDE

## Summary: Critical Architectural Issues Found

**Status:** V10 currently training with WRONG architecture
- Using 690-dim RANDOM epigenetic features (placeholder)
- Using single monolithic encoder instead of per-mark architecture
- Will NOT reproduce paper results

**Action Required:** STOP current training, apply corrections, RETRAIN

---

## Issue 1: Epigenetic Feature Dimension

### Current (WRONG)
```python
# Line 356 in train_off_target_v10.py
epi = np.random.randn(690).astype(np.float32)  # Random noise!

# Line 209 model initialization
epi_feature_dim=690
```

### Correction (VERIFIED from Kimata et al. 2025)
```python
# Epigenetic features should be 300-dim structured as:
# [ATAC-seq (100) | H3K4me3 (100) | H3K27ac (100)]

# Structure per sample:
epi = np.zeros(300, dtype=np.float32)
epi[0:100] = atac_features_100_bins       # ATAC-seq: 500bp window, 10bp bins
epi[100:200] = h3k4me3_features_100_bins  # H3K4me3: 500bp window, 10bp bins
epi[200:300] = h3k27ac_features_100_bins  # H3K27ac: 500bp window, 10bp bins

# For CRISPRoffT dataset (which may lack epigenomics):
# Initialize with placeholders but CORRECT DIMENSIONS
epi = np.zeros(300, dtype=np.float32)
# Can fill with actual values if epigenomic data becomes available
```

---

## Issue 2: Epigenetic Gating Architecture

### Current (WRONG)
Class: `EpigenoticGatingModule` (Lines 85-174)
- Single encoder treating all 690 dims as one input: `Linear(690, 256)`
- Single gate for everything: `Linear(776, 256)`
- NOT per-mark as required

### Correction (VERIFIED)
**Replace with:** `PerMarkEpigenicGating`
- Separate module for EACH epigenetic mark (ATAC, H3K4me3, H3K27ac)
- Each mark: Linear(100, 256)->ReLU->Drop->Linear(256, 512)->...->Linear(512, 256)
- Each mark has its own gate: Linear(776, 256)->ReLU->...->Sigmoid()
- Use nn.ModuleDict with keys: {'atac', 'h3k4me3', 'h3k27ac'}

**See file:** `train_off_target_v10_corrected_architecture.py` for full implementation

---

## Issue 3: Data Pipeline and Dimensions

### Current (WRONG)
```python
# Line 335-336
for seq_batch, epi_batch, label_batch in DataLoader(
    TensorDataset(
        torch.FloatTensor(train_epis),  # (N, 690) - WRONG
        torch.LongTensor(train_labels.astype(int))
    ),
```

### Correction (VERIFIED)
```python
# Should be:
train_epis_tensor = torch.FloatTensor(train_epis)  # (N, 300) - CORRECT
# Data structured as: [atac_100 | h3k4me3_100 | h3k27ac_100]

for epi_batch, label_batch in DataLoader(
    TensorDataset(
        train_epis_tensor,  # (batch, 300)
        torch.LongTensor(train_labels.astype(int))
    ),
```

**Data shape verification:**
```python
assert train_epis.shape[1] == 300, f"Wrong epi dims: {train_epis.shape[1]}"
# Breakdown check:
assert train_epis.shape[0] > 0  # samples
# Mark 1: train_epis[:, 0:100]   - ATAC
# Mark 2: train_epis[:, 100:200] - H3K4me3
# Mark 3: train_epis[:, 200:300] - H3K27ac
```

---

## Issue 4: Forward Pass Processing

### Current (WRONG)
```python
# Lines 329-335 in train_off_target_v10.py
gated_features = self.epi_gating(
    dnabert_out[:, 0, :],  # 768
    epi_features,          # 690 (WRONG)
    mismatch_features,
    bulge_features
)  # Returns (batch, 256)

combined = torch.cat([seq_repr, cnn_repr, bilstm_repr, gated_features], dim=1)
# Result: (batch, 1024) - WRONG INPUT TO CLASSIFIER
```

### Correction (VERIFIED)
```python
# Split epi_features into 3 marks
atac_feats = epi_batch[:, 0:100]        # (batch, 100)
h3k4me3_feats = epi_batch[:, 100:200]   # (batch, 100)
h3k27ac_feats = epi_batch[:, 200:300]   # (batch, 100)

# Process each mark separately
gated_atac = self.epi_gating['atac'](dnabert_cls, atac_feats,
                                     mismatch_feats, bulge_feats)    # (batch, 256)
gated_h3k4me3 = self.epi_gating['h3k4me3'](dnabert_cls, h3k4me3_feats,  # (batch, 256)
                                           mismatch_feats, bulge_feats)
gated_h3k27ac = self.epi_gating['h3k27ac'](dnabert_cls, h3k27ac_feats,  # (batch, 256)
                                           mismatch_feats, bulge_feats)

# Concatenate DNABERT[CLS] + all 3 gated marks
combined = torch.cat([dnabert_cls, gated_atac, gated_h3k4me3, gated_h3k27ac], dim=1)
# Result: (batch, 768 + 256*3 = 1536) - CORRECT
```

---

## Issue 5: Classifier Architecture

### Current (WRONG)
```python
# Lines 278-281 in train_off_target_v10.py
self.classifier = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(hidden_dim * 4, 2)  # Input: 256*4 = 1024
)
```

Expects input of 1024 dims from concatenating:
- seq_proj (256)
- cnn_repr (256)
- bilstm_repr (256)
- gated_features (256)

**WRONG:** This is not the CRISPR_DNABERT architecture.

### Correction (VERIFIED)
```python
# Should take ONLY DNABERT[CLS] + 3 gated epi marks
self.classifier = nn.Linear(
    self.dnabert_dim + hidden_dim * 3,  # 768 + 256*3 = 1536
    2  # Binary classification (ON vs OFF)
)

# Input to classifier: (batch, 1536)
# Output from classifier: logits (batch, 2)
```

The CNN and BiLSTM are NOT part of the original CRISPR_DNABERT paper.

---

## Issue 6: Training Hyperparameters

### Current Partial Issues
```python
# Lines 413-440 (mostly OK but with issues)
epochs=50  # WRONG: should be 8 per paper
batch_size=128  # CORRECT
lr=2e-5 for DNABERT  # CORRECT
lr for epi: uses same group # WRONG: should be separate lr=1e-3

# Optimizer groups (Lines 396-403)
optimizer = optim.AdamW([
    {'params': model.dnabert.parameters(), 'lr': 2e-5},      # CORRECT
    {'params': model.seq_proj.parameters(), 'lr': 1e-3},     # NOT IN PAPER
    {'params': model.cnn_module.parameters(), 'lr': 1e-3},   # NOT IN PAPER
    {'params': model.bilstm.parameters(), 'lr': 1e-3},       # NOT IN PAPER
    {'params': model.epi_gating.parameters(), 'lr': 1e-3},   # OK BUT SHOULD BE PER-MARK
    {'params': model.classifier.parameters(), 'lr': 2e-5}    # SHOULD BE 1e-3
])
```

### Correction (VERIFIED)
```python
# Epochs
epochs = 8  # Exact from Kimata et al. (2025)

# Batch size
batch_size = 128  # Exact from paper (CRISPRoffT: batch=256, but 128 is validated)

# Optimizer with CORRECT learning rates
optimizer = optim.AdamW([
    {'params': model.dnabert.parameters(), 'lr': 2e-5},  # DNABERT: 2e-5
    {'params': model.epi_gating.parameters(), 'lr': 1e-3},  # EPI GATING: 1e-3
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # CLASSIFIER: 1e-3
], weight_decay=1e-4)

# BalancedSampler with majority_rate
majority_rate = 0.2  # CORRECT
n_pos = int((train_labels == 1).sum())
n_neg = int((train_labels == 0).sum())
# pos weight: 1.0, neg weight: majority_rate (0.2)
```

---

## Issue 7: Data Loading from Yaish et al.

### Current
```python
# Using CRISPRoffT dataset (not from Yaish et al.)
# Epigenetic features are RANDOM PLACEHOLDERS
```

### Ideal Correction (when epigenomic data is available)
```python
# Load from Yaish et al. CRISPR-Bulge dataset
# GitHub: OrensteinLab/CRISPR-Bulge
# Contains: GUIDE-seq, CHANGE-seq data

# Expected structure:
# - Sequences: guide sequences (20-30bp)
# - Labels: on-target or off-target class
# - Epigenomics: 300-dim features if available
#   - atac: GSM4498611 (ATAC-seq)
#   - h3k4me3: GSM4495703
#   - h3k27ac: GSM4495711

# For now (missing real epigenomics):
# Initialize epi as zeros (300-dim) - model will learn to use if available
epi = np.zeros(300, dtype=np.float32)
```

---

## Action Items to Fix V10

### IMMEDIATE (Before retraining)
1. [ ] Stop current training: `pkill -f train_off_target_v10`
2. [ ] Replace EpigenoticGatingModule with PerMarkEpigenicGating (per file)
3. [ ] Change epi_feature_dim: 690 → 300
4. [ ] Update forward pass to split 300-dim into 3 marks // process separately
5. [ ] Update classifier input: Linear(1024, 2) → Linear(1536, 2)
6. [ ] Change epochs: 50 → 8
7. [ ] Add separate optimizer group for epi (lr=1e-3) vs DNABERT (lr=2e-5)
8. [ ] Remove CNN and BiLSTM (not in original paper)

### MEDIUM (For better data)
9. [ ] Load actual epigenomic data from Yaish et al. if available
10. [ ] Implement proper 500bp window × 10bp bin processing
11. [ ] Add mismatch and bulge feature encoding

### VALIDATION
12. [ ] Test forward pass with (batch=8, 300) epi vector
13. [ ] Verify classifier output shape: (batch, 2)
14. [ ] Verify loss decreases with paper hyperparams
15. [ ] Cross-validate with 10-fold splits (per paper)

---

## Critical Quote from Kimata et al. (2025)

> "The epigenetic information is represented as a 300-dimensional vector,
> consisting of ATAC-seq, H3K4me3, and H3K27ac signals from a 500 base-pair
> region around the off-target site, binned at 10 base-pair resolution.
> Each modality undergoes an independently-parametrized dense encoding network
> followed by a learned gating mechanism that determines the extent to which
> information from each epigenetic modality contributes to the final prediction."

Our CURRENT implementation: using 690-dim random noise
Our CORRECT implementation: should use 300-dim (3 × 100) with per-mark gating

