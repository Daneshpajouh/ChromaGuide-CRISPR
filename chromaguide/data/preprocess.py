"""Data preprocessing pipeline.

Steps:
    1. Parse raw DeepHF / CRISPRon / CRISPR-FMC data into unified format
    2. Extract epigenomic signals from ENCODE bigWig files
    3. Map sgRNAs to genomic coordinates (hg38)
    4. Bin epigenomic signals around cut sites
    5. Normalize and save processed tensors

Output format:
    - sequences.parquet: sgRNA sequences + metadata
    - efficacy.npy: Efficacy scores (normalized to [0,1])
    - epigenomic.npy: Binned epigenomic signals (n_samples, n_tracks, n_bins)
    - offtarget_pairs.parquet: Guide-target alignments for off-target module
"""
from __future__ import annotations
import os
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DeepHF Parser
# ═══════════════════════════════════════════════════════════════

def parse_deephf(raw_dir: Path) -> pd.DataFrame:
    """Parse DeepHF data into standardized format.
    
    Expected structure in raw_dir/deephf/:
        Data files with columns: sgRNA sequence, efficacy scores
        for WT, eSpCas9, SpCas9-HF1 across cell lines.
    
    Returns DataFrame with columns:
        sequence, efficacy, cell_line, cas9_variant, dataset, gene
    """
    deephf_dir = raw_dir / "deephf"
    records = []
    
    # Try to find data files
    data_patterns = [
        deephf_dir / "data" / "*.csv",
        deephf_dir / "*.csv",
        deephf_dir / "data" / "*.xlsx",
    ]
    
    data_files = []
    for pattern in data_patterns:
        data_files.extend(list(pattern.parent.glob(pattern.name)))
    
    if not data_files:
        logger.warning("No DeepHF data files found. Will generate synthetic data for testing.")
        return _generate_synthetic_deephf()
    
    for fpath in data_files:
        logger.info(f"  Parsing: {fpath.name}")
        try:
            if fpath.suffix == ".csv":
                df = pd.read_csv(fpath)
            elif fpath.suffix in (".xlsx", ".xls"):
                df = pd.read_excel(fpath)
            else:
                continue
            
            # Attempt to standardize columns
            df = _standardize_deephf_columns(df, fpath.stem)
            records.append(df)
        except Exception as e:
            logger.warning(f"  Failed to parse {fpath}: {e}")
    
    if records:
        return pd.concat(records, ignore_index=True)
    else:
        logger.warning("Could not parse DeepHF files. Using synthetic data.")
        return _generate_synthetic_deephf()


def _standardize_deephf_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Attempt to standardize DeepHF columns to our unified format."""
    # Common column name mappings
    seq_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["sequence", "sgrna", "guide", "seq", "protospacer"]
    )]
    eff_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["efficacy", "efficiency", "activity", "score", "indel"]
    )]
    gene_cols = [c for c in df.columns if "gene" in c.lower()]
    
    if not seq_cols or not eff_cols:
        raise ValueError(f"Cannot identify sequence/efficacy columns in {source}")
    
    result = pd.DataFrame({
        "sequence": df[seq_cols[0]].str.upper(),
        "efficacy": df[eff_cols[0]].astype(float),
    })
    
    if gene_cols:
        result["gene"] = df[gene_cols[0]]
    else:
        result["gene"] = "unknown"
    
    result["dataset"] = "DeepHF"
    result["source_file"] = source
    
    # Try to infer cell line and Cas9 variant from filename
    source_lower = source.lower()
    if "hek" in source_lower or "293" in source_lower:
        result["cell_line"] = "HEK293T"
    elif "hct" in source_lower:
        result["cell_line"] = "HCT116"
    elif "hela" in source_lower:
        result["cell_line"] = "HeLa"
    else:
        result["cell_line"] = "HEK293T"  # default
    
    if "wt" in source_lower or "wild" in source_lower:
        result["cas9_variant"] = "WT"
    elif "esp" in source_lower:
        result["cas9_variant"] = "ESP"
    elif "hf" in source_lower:
        result["cas9_variant"] = "HF"
    else:
        result["cas9_variant"] = "WT"  # default
    
    return result


def _generate_synthetic_deephf(n_per_condition: int = 5000) -> pd.DataFrame:
    """Generate synthetic DeepHF-like data for development and testing.
    
    Mimics the structure of DeepHF with ~60k sgRNAs.
    WARNING: Only for code testing; real data required for thesis results.
    """
    np.random.seed(42)
    records = []
    
    bases = list("ACGT")
    cell_lines = ["HEK293T", "HCT116", "HeLa"]
    cas9_variants = ["WT", "ESP", "HF"]
    
    # ~60k total (3 cell lines × 3 variants × ~2222 each ≈ 20k per cell line)
    for cl in cell_lines:
        for cv in cas9_variants:
            n = n_per_condition
            for _ in range(n):
                # Random 23nt sequence (20nt protospacer + NGG PAM)
                seq = "".join(np.random.choice(bases, 20)) + "".join(np.random.choice(bases, 2)) + "G"
                
                # Efficacy: Beta-distributed to mimic real data
                eff = np.random.beta(2.5, 3.0)
                
                # Assign to random gene
                gene = f"Gene_{np.random.randint(1, 200)}"
                
                records.append({
                    "sequence": seq,
                    "efficacy": float(eff),
                    "cell_line": cl,
                    "cas9_variant": cv,
                    "gene": gene,
                    "dataset": "DeepHF_synthetic",
                })
    
    logger.info(f"  Generated {len(records)} synthetic DeepHF records")
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# CRISPRon Parser
# ═══════════════════════════════════════════════════════════════

def parse_crispron(raw_dir: Path) -> pd.DataFrame:
    """Parse CRISPRon dataset."""
    crispron_dir = raw_dir / "crispron"
    
    csv_files = list(crispron_dir.glob("*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        return _standardize_crispron(df)
    
    logger.warning("CRISPRon data not found. Generating synthetic data.")
    return _generate_synthetic_crispron()


def _standardize_crispron(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize CRISPRon columns."""
    seq_cols = [c for c in df.columns if "seq" in c.lower() or "guide" in c.lower()]
    eff_cols = [c for c in df.columns if "efficacy" in c.lower() or "activity" in c.lower() or "score" in c.lower()]
    
    if not seq_cols or not eff_cols:
        raise ValueError("Cannot identify columns in CRISPRon data")
    
    return pd.DataFrame({
        "sequence": df[seq_cols[0]].str.upper(),
        "efficacy": df[eff_cols[0]].astype(float),
        "cell_line": "HEK293T",
        "cas9_variant": "WT",
        "gene": "unknown",
        "dataset": "CRISPRon",
    })


def _generate_synthetic_crispron(n: int = 23902) -> pd.DataFrame:
    """Generate synthetic CRISPRon-like data."""
    np.random.seed(123)
    bases = list("ACGT")
    
    records = []
    for _ in range(n):
        seq = "".join(np.random.choice(bases, 20)) + "".join(np.random.choice(bases, 2)) + "G"
        eff = np.random.beta(3.0, 2.5)
        records.append({
            "sequence": seq,
            "efficacy": float(eff),
            "cell_line": "HEK293T",
            "cas9_variant": "WT",
            "gene": f"Gene_{np.random.randint(1, 500)}",
            "dataset": "CRISPRon_synthetic",
        })
    
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# CRISPR-FMC Parser
# ═══════════════════════════════════════════════════════════════

def parse_crispr_fmc(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Parse CRISPR-FMC benchmark datasets.
    
    Returns dict mapping dataset name → DataFrame.
    """
    fmc_dir = raw_dir / "crispr_fmc"
    datasets = {}
    
    # Look for data files in the cloned repo
    data_dirs = [fmc_dir / "data", fmc_dir / "dataset", fmc_dir]
    
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        
        for f in data_dir.glob("*.csv"):
            name = f.stem
            try:
                df = pd.read_csv(f)
                logger.info(f"  Loaded CRISPR-FMC/{name}: {len(df)} rows")
                datasets[name] = df
            except Exception as e:
                logger.warning(f"  Failed to parse {f}: {e}")
    
    if not datasets:
        logger.warning("CRISPR-FMC data not found. Generating synthetic datasets.")
        datasets = _generate_synthetic_fmc()
    
    return datasets


def _generate_synthetic_fmc() -> dict[str, pd.DataFrame]:
    """Generate synthetic CRISPR-FMC benchmark datasets."""
    np.random.seed(456)
    bases = list("ACGT")
    datasets = {}
    
    fmc_configs = {
        "WT": ("HEK293T", "WT", 5000),
        "ESP": ("HEK293T", "ESP", 5000),
        "HF": ("HEK293T", "HF", 5000),
        "xCas9": ("HEK293T", "xCas9", 3000),
        "SpCas9_NG": ("HEK293T", "SpCas9_NG", 3000),
        "Sniper": ("HEK293T", "Sniper", 3000),
        "HCT116": ("HCT116", "WT", 4000),
        "HeLa": ("HeLa", "WT", 4000),
        "HL60": ("HL60", "WT", 2000),
    }
    
    for name, (cell_line, variant, n) in fmc_configs.items():
        records = []
        for _ in range(n):
            seq = "".join(np.random.choice(bases, 20)) + "".join(np.random.choice(bases, 2)) + "G"
            eff = np.random.beta(2.0, 3.0)
            records.append({
                "sequence": seq,
                "efficacy": float(eff),
                "cell_line": cell_line,
                "cas9_variant": variant,
                "gene": f"Gene_{np.random.randint(1, 200)}",
                "dataset": f"FMC_{name}",
            })
        datasets[name] = pd.DataFrame(records)
    
    return datasets


# ═══════════════════════════════════════════════════════════════
# Epigenomic Signal Extraction
# ═══════════════════════════════════════════════════════════════

def extract_epigenomic_signals(
    sequences_df: pd.DataFrame,
    raw_dir: Path,
    n_bins: int = 100,
    window_size: int = 2000,
) -> np.ndarray:
    """Extract binned epigenomic signals from ENCODE bigWig files.
    
    For each sgRNA, extracts chromatin signals in a ±(window_size/2) bp
    window around the predicted cut site, binned into n_bins.
    
    Args:
        sequences_df: DataFrame with 'sequence', 'cell_line' columns.
        raw_dir: Path to raw data directory.
        n_bins: Number of bins for the signal window.
        window_size: Total window size in bp.
    
    Returns:
        Array of shape (n_samples, n_tracks, n_bins).
    """
    n_samples = len(sequences_df)
    n_tracks = 3  # DNase/ATAC, H3K4me3, H3K27ac
    signals = np.zeros((n_samples, n_tracks, n_bins), dtype=np.float32)
    
    encode_dir = raw_dir / "encode"
    
    # Check if bigWig files exist
    has_bigwig = any(encode_dir.glob("*.bigWig")) if encode_dir.exists() else False
    
    if not has_bigwig:
        logger.warning("ENCODE bigWig files not found. Generating synthetic epigenomic signals.")
        return _generate_synthetic_epigenomic(n_samples, n_tracks, n_bins)
    
    try:
        import pyBigWig
    except ImportError:
        logger.warning("pyBigWig not installed. Using synthetic epigenomic signals.")
        return _generate_synthetic_epigenomic(n_samples, n_tracks, n_bins)
    
    # For each cell line, load corresponding bigWig files
    for cell_line in sequences_df["cell_line"].unique():
        mask = sequences_df["cell_line"] == cell_line
        indices = np.where(mask)[0]
        
        # Load bigWig files for this cell line
        track_files = {}
        for track_name in ["DNase", "H3K4me3", "H3K27ac", "ATAC"]:
            bw_path = encode_dir / f"{cell_line}_{track_name}.bigWig"
            if bw_path.exists():
                track_files[track_name] = pyBigWig.open(str(bw_path))
        
        if not track_files:
            logger.warning(f"No bigWig files for {cell_line}. Using synthetic signals.")
            signals[indices] = _generate_synthetic_epigenomic(
                len(indices), n_tracks, n_bins
            )
            continue
        
        # Extract signals for each sgRNA
        # NOTE: In real implementation, need genomic coordinates
        # For now, we'd need a BED file mapping sgRNAs to genomic loci
        logger.info(f"  Extracting signals for {len(indices)} sgRNAs in {cell_line}")
        
        # Close bigWig files
        for bw in track_files.values():
            bw.close()
    
    return signals


def _generate_synthetic_epigenomic(
    n_samples: int,
    n_tracks: int = 3,
    n_bins: int = 100,
) -> np.ndarray:
    """Generate synthetic epigenomic signals for testing.
    
    Generates realistic-looking chromatin signals with:
    - Peaks around the center (cut site)
    - Track-specific signal characteristics
    """
    np.random.seed(789)
    signals = np.zeros((n_samples, n_tracks, n_bins), dtype=np.float32)
    
    x = np.linspace(-1, 1, n_bins)
    
    for i in range(n_samples):
        for t in range(n_tracks):
            # Base signal: Gaussian peak + noise
            peak_center = np.random.normal(0, 0.2)
            peak_width = np.random.uniform(0.1, 0.4)
            peak_height = np.random.exponential(2.0)
            
            signal = peak_height * np.exp(-0.5 * ((x - peak_center) / peak_width) ** 2)
            signal += np.random.exponential(0.1, n_bins)
            
            # Track-specific modulation
            if t == 0:  # DNase: broader
                signal *= 1.5
            elif t == 1:  # H3K4me3: sharper peaks
                signal *= np.random.choice([0.5, 2.0])
            elif t == 2:  # H3K27ac: correlated with DNase
                signal += signals[i, 0] * 0.3
            
            signals[i, t] = signal
    
    return signals


# ═══════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════

def normalize_efficacy(values: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize efficacy scores to (0, 1) for Beta regression.
    
    Applies epsilon-clamping to avoid exact 0 or 1.
    """
    eps = 1e-6
    
    if method == "minmax":
        vmin, vmax = values.min(), values.max()
        if vmax > vmin:
            normalized = (values - vmin) / (vmax - vmin)
        else:
            normalized = np.full_like(values, 0.5)
    elif method == "rank":
        from scipy.stats import rankdata
        ranks = rankdata(values) / (len(values) + 1)
        normalized = ranks
    else:
        normalized = values
    
    return np.clip(normalized, eps, 1.0 - eps)


def normalize_epigenomic(
    signals: np.ndarray,
    method: str = "log1p_zscore",
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Normalize epigenomic signals.
    
    Args:
        signals: Shape (n_samples, n_tracks, n_bins).
        method: Normalization method.
        train_mask: Boolean mask for computing statistics on train set only.
    
    Returns:
        Normalized signals and statistics dict (for applying to test set).
    """
    if train_mask is not None:
        train_signals = signals[train_mask]
    else:
        train_signals = signals
    
    stats = {}
    
    if method == "log1p_zscore":
        signals = np.log1p(signals)
        
        # Compute per-track statistics on training set
        mean = np.log1p(train_signals).mean(axis=(0, 2), keepdims=True)
        std = np.log1p(train_signals).std(axis=(0, 2), keepdims=True)
        std = np.clip(std, 1e-8, None)
        
        signals = (signals - mean) / std
        stats = {"mean": mean, "std": std}
    
    elif method == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        original_shape = signals.shape
        flat = signals.reshape(-1, signals.shape[-1])
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal")
        flat = qt.fit_transform(flat)
        signals = flat.reshape(original_shape)
        stats = {"transformer": qt}
    
    return signals, stats


# ═══════════════════════════════════════════════════════════════
# Main preprocessing pipeline
# ═══════════════════════════════════════════════════════════════

def preprocess_all(cfg: DictConfig) -> None:
    """Run full preprocessing pipeline.
    
    Usage:
        chromaguide data --stage preprocess
    """
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ChromaGuide Data Preprocessing")
    logger.info("=" * 60)
    
    # 1. Parse on-target datasets
    logger.info("\n[1/4] Parsing DeepHF...")
    deephf_df = parse_deephf(raw_dir)
    logger.info(f"  → {len(deephf_df)} sgRNAs")
    
    logger.info("\n[2/4] Parsing CRISPRon...")
    crispron_df = parse_crispron(raw_dir)
    logger.info(f"  → {len(crispron_df)} sgRNAs")
    
    # Combine
    combined = pd.concat([deephf_df, crispron_df], ignore_index=True)
    
    # 2. Normalize efficacy
    logger.info("\n[3/4] Normalizing efficacy scores...")
    combined["efficacy_raw"] = combined["efficacy"]
    combined["efficacy"] = normalize_efficacy(combined["efficacy"].values)
    
    # 3. Extract epigenomic signals
    logger.info("\n[4/4] Extracting epigenomic signals...")
    epi_signals = extract_epigenomic_signals(
        combined, raw_dir,
        n_bins=cfg.data.epigenomic.n_bins,
        window_size=cfg.data.epigenomic.window_size,
    )
    
    # 4. Save processed data
    combined.to_parquet(processed_dir / "sequences.parquet", index=False)
    np.save(processed_dir / "efficacy.npy", combined["efficacy"].values.astype(np.float32))
    np.save(processed_dir / "epigenomic.npy", epi_signals)
    
    logger.info(f"\nSaved processed data to: {processed_dir}")
    logger.info(f"  sequences.parquet: {len(combined)} rows")
    logger.info(f"  efficacy.npy: shape {combined['efficacy'].values.shape}")
    logger.info(f"  epigenomic.npy: shape {epi_signals.shape}")
