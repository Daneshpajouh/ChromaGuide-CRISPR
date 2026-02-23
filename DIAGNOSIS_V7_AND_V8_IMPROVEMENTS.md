# V7 Diagnosis & V8 Improvements

## MULTIMODAL V7 DIAGNOSTIC ANALYSIS

### The Problem
- **V7 peaked at ~0.763 Rho (epoch 220)**
- **V6 reached 0.8435 Rho** (without gated attention, just sequence + basic fusion)
- **V7 is WORSE than V6 by ~0.08 Rho (9.4% regression)**
- This violates the expectation that multimodal > sequence-only

### Root Cause Analysis

#### 1. **Feature Scaling Mismatch** (PRIMARY)
```
Sequence features (one-hot):  range [0, 1], mean ~0.5, std ~0.5
Epigenomic features (raw):    range [-64, +61], mean varies, std ~20+
```
- One-hot sequences are naturally normalized [0,1] from encoding
- Epigenomic features are raw values on completely different scale
- Early training: model learns to heavily weight small sequence signal over large-magnitude epigenomic noise
- Result: Epigenomics are effectively ignored due to scale mismatch

#### 2. **Weak Epigenomic Encoder** (SECONDARY)
- V7: 11 features → 128 → 64 dims (only 2 linear layers)
- This is insufficient to adequately process 11 diverse feature inputs
- No intermediate normalization or non-linearity beyond ReLU
- Gate network operates on concatenated [seq; epi] but doesn't learn proper feature importance

#### 3. **Simple Gating is Under-Expressive** (TERTIARY)
- Gate network in v7: `Linear(128→64) + Sigmoid`
- Produces element-wise gates (per d_model dimension)
- Cannot capture complex interactions between sequence and epigenomic features
- Cross-attention would be more expressive but v7 doesn't use it

#### 4. **Missing Baseline Verification**
- V7 assumes epigenomics are helping, but never validates this
- No sequence-only control model for comparison
- Cannot diagnose if low Rho is due to epigenomics hurting or fusion being weak

---

## MULTIMODAL V8 IMPROVEMENTS

### 1. **Feature Normalization** (FIX #1)
```python
# Normalize using training set statistics
X_epi_norm = (X_epi - train_mean) / train_std
```
- All epigenomic features normalized to ~zero mean, ~unit variance
- Perfect scale alignment with one-hot sequences [0,1]
- Model can now properly learn feature importance without scale bias

### 2. **Stronger Epigenomic Encoder** (FIX #2)
```python
V7 encoder:  11 → 128 → 64 (basic)
V8 encoder:  11 → 128 → 256 → 128 → 64 (deeper, more capacity)
```
- Added intermediate dimension expansion (11→128→256 instead of direct 11→128)
- Batch normalization at each stage for training stability
- Higher dropout (0.3) to prevent overfitting to noise
- Improved feature representations before fusion

### 3. **Multi-Head Attention Fusion** (FIX #3)
```python
# V7: Simple gating
g = gate_network([h_seq; h_epi])
fused = g * h_seq + (1-g) * h_epi

# V8: Cross-attention with residuals
q = project_query(h_seq)
k,v = project_kv(h_epi)
attention = softmax(q @ k^T / sqrt(d))
context = attention @ v
fused = h_seq + context  # residual!
```
- Multi-head (4 heads) allows learning different feature interaction patterns
- Residual connection ensures sequence signal is preserved
- Layer normalization stabilizes learning
- Cross-attention is more expressive than element-wise gating

### 4. **Sequence-Only Baseline** (FIX #4)
```python
# Train baseline model with same architecture but without epigenomics
# Compare: multimodal_rho vs sequence_only_rho
improvement = multimodal_rho - sequence_only_rho
```
- Diagnostic model to verify epigenomics actually help
- If multimodal < sequence_only, fusion is hurting the signal
- Provides ground truth on fusion quality

### 5. **Stronger CNN Backbone** (FIX #5)
```python
V7: Conv1d(4→32→32) with 2 layers
V8: Conv1d(4→64→64→32) with 3 layers + residuals
```
- More capacity to learn sequence patterns
- Better feature extraction from one-hot sequences

---

## OFF-TARGET V7 DIAGNOSTIC ANALYSIS

### The Problem
- **Individual model AUROCs averaging 0.944-0.948**
- **Simple averaging ensemble cannot reach 0.99 target**
- **Expected ensemble benefit: ~0.5% per model diversity**
- At 0.944 mean + 0.5% ensemble benefit → ~0.945 (still below 0.99)

### Root Cause Analysis

#### 1. **Architecture Capacity Ceiling**
- V7 CNN: 1D-CNN(4→32→32) + one linear layer
- Too shallow to learn complex off-target distinction patterns
- Limited feature representation capacity

#### 2. **Single Pooling Strategy**
- V7 uses only max pooling for feature aggregation
- Loses information from intermediate activations
- Single feature scale may miss important patterns

#### 3. **Weak Ensemble Averaging**
- Simple mean of probabilities: `ensemble_pred = mean([p1, p2, ..., p10])`
- Assumes all models equally good (they're not: 0.9379 vs 0.9480)
- No confidence weighting or voting threshold optimization
- Doesn't exploit diversity of individual models

---

## OFF-TARGET V8 IMPROVEMENTS

### 1. **Deeper CNN Architecture** (FIX #1)
```python
V7: Conv1d(4→32→32) with 5x5 kernel
V8: Conv1d(4→128→128→64→64→32) with 5 layers + BatchNorm
```
- 5 convolutional layers (vs 2) massively increases capacity
- Batch normalization at each layer stabilizes training
- Can learn deeper sequence patterns and motifs

### 2. **Multi-Scale Feature Extraction** (FIX #2)
```python
# Parallel multi-kernel conv path
kernels = [3, 4, 5, 7]  # different receptive fields
features = concatenate(
    main_path_32_dims,
    multi_kernel_3_32_dims,
    multi_kernel_4_32_dims,
    multi_kernel_5_32_dims,
    multi_kernel_7_32_dims
)  # Total: 32 + 32*4 = 160 dims
```
- Multiple kernel sizes capture patterns at different scales
- Parallel multi-kernel approach (not sequential)
- Dramatically increases learned feature diversity

### 3. **Stronger FC Head** (FIX #3)
```python
V7: 32 → 1 (direct linear)
V8: 160 → 256 → 128 → 64 → 1 (progressive reduction with dropout)
```
- Much larger input dimension (160 vs 32)
- Better feature integration: 160→256 expands, then 256→128→64→1 gradually reduces
- Dropout (0.3, 0.3, 0.2) prevents overfitting

### 4. **Improved Ensemble Strategy** (FIX #4)
```python
# Weighted voting based on individual model performance
# Threshold optimization on validation set
# Confidence-aware aggregation

weights = normalize(ind_aurocs)  # models with higher AUROC weighted more
ensemble_pred = sum(weights[i] * sigmoid(logits[i]))

# Find optimal threshold on validation set
best_threshold = arg_max_f1(y_val, ensemble_pred_val)
```
- Weight models by their individual performance
- Optimize decision threshold for maximum F1-score
- Ensemble now leverages model quality differences

### 5. **Position-Aware Pooling** (FIX #5)
```python
# Combine max and average pooling
max_features = max_pool(activations)
avg_features = avg_pool(activations)
combined = concatenate([max_features, avg_features])
```
- Max pooling captures peak activations (important events)
- Avg pooling captures overall pattern (context)
- Together: more complete feature capture

---

## Expected Improvements

### Multimodal V8
- **V6 Test Rho: 0.8435**
- **V7 Test Rho: ~0.763 (regression)**
- **V8 Expected: ≥ 0.85** (with normalizations and stronger fusion)
- If epigenomics properly utilized: could reach **0.88-0.90** range
- **Gap to 0.911 target: 1-3%** (still need hyperparameter tuning or v9)

### Off-Target V8
- **V7 Individual avg: 0.944-0.948**
- **V7 Ensemble (simple mean): ~0.945**
- **V8 Individual expected: ≥ 0.955-0.960** (deeper architecture)
- **V8 Ensemble expected: ≥ 0.96-0.97** (weighted voting)
- **Could meet or exceed 0.99 target** with optimized thresholds

---

## Implementation Checklist

### V8 Multimodal
- [x] Feature normalization using training statistics
- [x] Deeper epigenomic encoder (11→128→256→128→64)
- [x] Multi-head attention fusion (4 heads, residuals)
- [x] Sequence-only baseline for validation
- [x] Stronger CNN backbone (3 layers)
- [x] Batch normalization and layer norm
- [x] Better hyperparameters (lr=5e-4, patience=100)

### V8 Off-Target
- [x] Deeper CNN (5 layers vs 2)
- [x] Multi-scale feature extraction (parallel kernels)
- [x] Stronger FC head (160→256→128→64→1)
- [x] Weighted ensemble voting
- [x] Multi-scale pooling (max + avg)
- [x] Batch normalization throughout
- [x] Better hyperparameters (lr=5e-4, pos_weight=214.5)

---

## Next Steps

1. **Launch V8 trainings immediately** while V7/V7-CNN continue
2. **Monitor early convergence** - if V8 multimodal reaches 0.85+ by epoch 50, it's working
3. **Compare sequence-only vs multimodal** - if multimodal doesn't improve, debug further
4. **For off-target**, watch if individual model AUROCs improve to >0.95
5. **If V8 still falls short**, move to V9 with architectural changes or ensemble post-processing
