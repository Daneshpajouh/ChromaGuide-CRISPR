# ChromaGuide v2 — Bug Report (Full Audit)

## Date: 2026-02-26

## CRITICAL BUGS

### Bug 1: `total_mem` → `total_memory` in train_experiment.py (FIXED)
- **File**: `experiments/train_experiment.py` line 340
- **Issue**: `torch.cuda.get_device_properties(0).total_mem` should be `.total_memory`
- **Impact**: Crashes immediately on any GPU cluster
- **Status**: FIXED

### Bug 2: Missing `--backbone` arg in redistributed SLURM scripts (FIXED)
- **Files**: 15 redistributed SLURM scripts in `experiments/slurm_jobs/`
- **Impact**: Jobs fail to start — argparse requires `--backbone`
- **Status**: FIXED

### Bug 3: `total_mem` → `total_memory` in reproducibility.py (NEW)
- **File**: `chromaguide/utils/reproducibility.py` line 25
- **Issue**: Same attribute name bug as Bug 1
- **Impact**: Crashes when `get_device()` is called on GPU

### Bug 4: Raw sequences not passed to transformer-based encoders (NEW — CRITICAL)
- **File**: `experiments/train_experiment.py` — `train_epoch()` and `evaluate()`
- **Issue**: The training loop calls `model(seq, epi)` without passing `raw_sequences`. For transformer-based backbones (DNABERT-2, Evo, NT), this means:
  - If pretrained model IS loaded: crashes with ValueError
  - If pretrained model NOT loaded (fallback): runs with random features (garbage results)
- **Root cause**: `CRISPRDataset.__getitem__()` returns `sequence_str` in the batch, but the training script ignores it
- **Impact**: ALL transformer backbone experiments (30 of 45 jobs) will either crash or produce meaningless results

### Bug 5: `nucleotide_transformer` missing from `_needs_raw_sequences` (NEW — CRITICAL)
- **File**: `chromaguide/models/chromaguide.py` line 70
- **Issue**: `_needs_raw_sequences` only checks for `["dnabert2", "evo"]`, not `"nucleotide_transformer"`
- **Impact**: Even if raw_sequences were passed, NT encoder wouldn't receive them

### Bug 6: Missing `NucleotideTransformerEncoder` in __init__.py (MINOR)
- **File**: `chromaguide/modules/__init__.py`
- **Issue**: Not exported (only cosmetic, build_sequence_encoder still works)

## POTENTIAL ISSUES (Non-blocking)

### Issue 7: `rich` dependency in logging.py
- The `setup_logger()` imports `rich.logging.RichHandler` which may not be installed
- **Mitigation**: train_experiment.py uses its own logging setup; this is only hit via package imports

### Issue 8: Deprecated `torch.cuda.amp` imports in trainer.py  
- Uses `from torch.cuda.amp import autocast, GradScaler` (deprecated path)
- **Mitigation**: Still works on current PyTorch versions used on Alliance clusters
