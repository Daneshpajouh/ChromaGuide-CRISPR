# V10 ARCHITECTURE CORRECTION - IMPLEMENTATION PLAN

**Status:** TRAINING STOPPED - Architecture has critical errors
**Date:** February 23, 2026, 1:45 PM
**Priority:** CRITICAL - Must fix before retraining

---

## Summary of Issues Found

### Issue 1: Epigenetic Feature Dimension (CRITICAL ❌)
- **Current:** 690-dimensional RANDOM NOISE
- **Correct:** 300-dimensional (3 marks × 100 bins each)
- **Source:** Kimata et al. (2025), PLOS ONE
- **Impact:** Model learning from worthless random features
- **Fix complexity:** MEDIUM

### Issue 2: Epigenetic Module Architecture (CRITICAL ❌)
- **Current:** Single monolithic encoder `Linear(690, 256)`
- **Correct:** nn.ModuleDict with 3 separate encoders
  - 'atac': Linear(100, 256)→ReLU→Drop→...→Linear(512, 256)
  - 'h3k4me3': Linear(100, 256)→ReLU→Drop→...→Linear(512, 256)
  - 'h3k27ac': Linear(100, 256)→ReLU→Drop→...→Linear(512, 256)
- **Impact:** Wrong mathematical structure
- **Fix complexity:** MEDIUM-HIGH

### Issue 3: Classifier Input Dimension (CRITICAL ❌)
- **Current:** `Linear(1024, 2)` from [seq_proj|cnn_repr|bilstm_repr|gated_epi]
- **Correct:** `Linear(1536, 2)` from [dnabert_cls|gated_atac|gated_h3k4me3|gated_h3k27ac]
- **Impact:** Classifier can't be trained without fixing input
- **Fix complexity:** LOW

### Issue 4: Extra Components Not in Paper (IMPORTANT ⚠️)
- **Current:** Includes CNN (MultiScaleCNNModule) and BiLSTM (BiLSTMContext)
- **Correct:** Only DNABERT + per-mark epigenetic gating
- **Source:** Kimata et al. (2025) architecture
- **Impact:** Not comparable to published results
- **Fix complexity:** MEDIUM

### Issue 5: Training Epochs (IMPORTANT ⚠️)
- **Current:** epochs=50 (extended for diversity)
- **Correct:** epochs=8 (exact from paper)
- **Impact:** May cause overfitting
- **Fix complexity:** LOW

### Issue 6: Optimizer Learning Rate Groups (IMPORTANT ⚠️)
- **Current:** 
  - DNABERT: 2e-5 ✓
  - Other params: scatter across multiple groups
- **Correct:**
  - DNABERT: 2e-5
  - Epi gating (all 3 marks): 1e-3
  - Classifier: 1e-3
- **Impact:** Different convergence rates
- **Fix complexity:** LOW

---

## Detailed Changes Required

### File: `/Users/studio/Desktop/PhD/Proposal/scripts/train_off_target_v10.py`

#### Change 1: Replace EpigenoticGatingModule (Lines 85-174)
**Delete entire class** and replace with `PerMarkEpigenicGating` class

**Reference implementation:**
- See `train_off_target_v10_corrected_architecture.py` for complete code

**Key aspects:**
- Input: mark_dim=100 (not 690!)
- Output: hidden_dim=256 per mark
- Gate takes: DNABERT(768) + mismatch(7) + bulge(1) = 776 dims
- Gate outputs: scalar (0-1) to weight the encoded features
- Initialize gate bias to -3.0

#### Change 2: Update Model Init (Line 209)
```python
# Change from:
epi_feature_dim=690

# To:
epi_feature_dim=300  # NOT USED ANYMORE - will remove in refactor
```

#### Change 3: Update EPI Gating Creation (Lines 268-276)
```python
# Change from:
self.epi_gating = EpigenoticGatingModule(...)

# To:
self.epi_gating = nn.ModuleDict({
    'atac': PerMarkEpigenicGating(mark_dim=100, hidden_dim=256,
                                 dnabert_dim=self.dnabert_dim, dropout=dropout),
    'h3k4me3': PerMarkEpigenicGating(mark_dim=100, hidden_dim=256,
                                    dnabert_dim=self.dnabert_dim, dropout=dropout), 
    'h3k27ac': PerMarkEpigenicGating(mark_dim=100, hidden_dim=256,
                                    dnabert_dim=self.dnabert_dim, dropout=dropout)
})
```

#### Change 4: Remove CNN and BiLSTM (Lines 247-265, 266-267)
**DELETE these lines:**
```python
        # 2. Multi-scale CNN Module (CRISPR-MCA style)
        self.cnn_module = MultiScaleCNNModule(...)
        self.cnn_proj = nn.Sequential(...)

        # 3. BiLSTM for context (CRISPR-HW style)
        self.bilstm = BiLSTMContext(...)
        self.bilstm_proj = nn.Sequential(...)
```

#### Change 5: Update Classifier (Line 281)
```python
# Change from:
self.classifier = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(hidden_dim * 4, 2)  # 1024 input
)

# To:
self.classifier = nn.Linear(
    self.dnabert_dim + 256 * 3,  # 768 + 256*3 = 1536
    2
)
```

#### Change 6: Rewrite Forward Method (Lines 289-336)
**Complete rewrite:**
```python
def forward(self, sequences, epi_features, mismatch_features=None, bulge_features=None):
    # Tokenize and encode
    tokens = self.tokenizer(sequences, return_tensors="pt", padding=True,
                           truncation=True, max_length=24).to(device)
    dnabert_out = self.dnabert(**tokens).last_hidden_state
    dnabert_cls = dnabert_out[:, 0, :]  # (batch, 768)
    
    # Split 300-dim epi features
    atac_feats = epi_features[:, 0:100]       # (batch, 100)
    h3k4me3_feats = epi_features[:, 100:200]  # (batch, 100)
    h3k27ac_feats = epi_features[:, 200:300]  # (batch, 100)
    
    # Process each mark
    gated_atac = self.epi_gating['atac'](dnabert_cls, atac_feats,
                                         mismatch_features, bulge_features)
    gated_h3k4me3 = self.epi_gating['h3k4me3'](dnabert_cls, h3k4me3_feats,
                                               mismatch_features, bulge_features)
    gated_h3k27ac = self.epi_gating['h3k27ac'](dnabert_cls, h3k27ac_feats,
                                               mismatch_features, bulge_features)
    
    # Concatenate and classify
    combined = torch.cat([dnabert_cls, gated_atac, gated_h3k4me3, gated_h3k27ac], dim=1)
    logits = self.classifier(combined)
    
    return logits
```

#### Change 7: Update Data Loading (Line 356)
```python
# Change from:
epi = np.random.randn(690).astype(np.float32)

# To:
epi = np.zeros(300, dtype=np.float32)
# [0:100] = ATAC-seq features (when available)
# [100:200] = H3K4me3 features (when available)
# [200:300] = H3K27ac features (when available)
```

#### Change 8: Update Optimizer (Lines 396-403)
```python
# Change from:
optimizer = optim.AdamW([
    {'params': model.dnabert.parameters(), 'lr': 2e-5},
    {'params': model.seq_proj.parameters(), 'lr': 1e-3},
    {'params': model.cnn_module.parameters(), 'lr': 1e-3},
    {'params': model.bilstm.parameters(), 'lr': 1e-3},
    {'params': model.epi_gating.parameters(), 'lr': 1e-3},
    {'params': model.classifier.parameters(), 'lr': 2e-5}
], weight_decay=1e-4)

# To:
optimizer = optim.AdamW([
    {'params': model.dnabert.parameters(), 'lr': 2e-5},
    {'params': model.epi_gating.parameters(), 'lr': 1e-3},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)
```

#### Change 9: Fix Epochs (Line 529)
```python
# Change from:
epochs=50

# To:
epochs=8  # Exact from Kimata et al. (2025)
```

---

## Classes to Add/Remove

### Add (New)
```python
class PerMarkEpigenicGating(nn.Module):
    """See train_off_target_v10_corrected_architecture.py for full implementation"""
```

### Remove (Not in original paper)
```python
class MultiScaleCNNModule(nn.Module):  # REMOVE
class BiLSTMContext(nn.Module):  # REMOVE
```

---

## Testing Checklist Before Retrain

- [ ] Verify data shape: `assert X_epis.shape == (N, 300)`
- [ ] Verify epi structure: 
  - [ ] `X_epis[:, 0:100]` is ATAC-seq (100 dims)
  - [ ] `X_epis[:, 100:200]` is H3K4me3 (100 dims)
  - [ ] `X_epis[:, 200:300]` is H3K27ac (100 dims)
- [ ] Verify model forward pass works:
  ```python
  model = DNABERTOffTargetV10()
  test_seqs = ["ACGTACGTACGTACGTACGTACGT"]  # 24bp
  test_epi = torch.zeros(1, 300)
  output = model(test_seqs, test_epi)
  assert output.shape == (1, 2)  # Binary classification
  ```
- [ ] Verify optimizer has exactly 3 param groups
- [ ] Verify classifier input size is 1536 (768 + 256*3)
- [ ] Verify no CNN or BiLSTM modules remain
- [ ] Verify epochs is exactly 8

---

## Implementation Timeline

**Estimated effort:**
- Code changes: 1-2 hours (60 lines total)
- Testing and validation: 1 hour
- First retrain: 12-18 hours (5 models × 8 epochs)

**Total timeline:**
- Now (1:45 PM): Fix code
- 4:00 PM: Start corrected training
- Feb 24, 4:00 AM: Training complete
- Feb 24, 4:30 AM: Results evaluation

---

## Alternative: Quick V10 Fix Script

Instead of manually editing, could create a script to:
1. Read current V10
2. Extract per-mark logic
3. Replace class definitions
4. Update all references
5. Verify output

**Would be faster for future versions but slower for this urgent case**

---

## Validation Against Paper

Once corrected, validate:
- ✓ Epigenetic dimensions: 300 (3×100)
- ✓ Per-mark gating: yes (nn.ModuleDict)
- ✓ Batch size: 128
- ✓ Epochs: 8
- ✓ DNABERT lr: 2e-5
- ✓ Other lr: 1e-3
- ✓ Classifier input: 1536 (768+256×3)
- ✓ No extra components: CNN/BiLSTM removed

---

## Questions to Address After Fix

1. **Will accuracy match paper?**
   - Paper target AUROC: 0.99
   - Realistic expectation: 0.93-0.96
   - Depends on epigenomic data quality

2. **What's the epigenomic data source?**
   - Currently: zeros (placeholder)
   - Should be: GSM4498611 (ATAC), GSM4495703 (H3K4me3), GSM4495711 (H3K27ac)
   - Can load later without retraining architecture

3. **Does 8 epochs converge well?**
   - Yes, per paper
   - Have early stopping in place (patience=15)
   - Can run ensemble (5 models × 8 epochs = better diversity)

---

## Success Criteria

✅ Code passes syntax check  
✅ Model forward pass works  
✅ Dimensions match: input (batch,300) → output (batch,2)  
✅ Loss decreases during training  
✅ Validation AUROC ≥ 0.90 (reasonable)  
✅ Results reproducible (same seed)  
✅ Architecture matches Kimata et al. (2025)  
✅ Publication ready  

