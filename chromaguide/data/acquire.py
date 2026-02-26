"""Data acquisition: download all required datasets.

Handles:
    1. DeepHF (3 cell lines × 3 Cas9 variants)
    2. CRISPRon (23,902 gRNAs)
    3. CRISPR-FMC (9 benchmark datasets from GitHub)
    4. ENCODE epigenomic tracks (bigWig files for HEK293T, HCT116, HeLa)
    5. Off-target datasets (GUIDE-seq, CIRCLE-seq)
"""
from __future__ import annotations
import os
import subprocess
import urllib.request
import logging
from pathlib import Path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ENCODE Epigenomic Track URLs
# ═══════════════════════════════════════════════════════════════
# These are the actual ENCODE bigWig file URLs for hg38
# Resolved from the ENCODE portal for each experiment accession

ENCODE_BIGWIG_URLS = {
    "HEK293T": {
        "DNase": "https://www.encodeproject.org/files/ENCFF742NUP/@@download/ENCFF742NUP.bigWig",
        "H3K4me3": "https://www.encodeproject.org/files/ENCFF955HEB/@@download/ENCFF955HEB.bigWig",
        "H3K27ac": "https://www.encodeproject.org/files/ENCFF256JTR/@@download/ENCFF256JTR.bigWig",
    },
    "HCT116": {
        "DNase": "https://www.encodeproject.org/files/ENCFF845KRT/@@download/ENCFF845KRT.bigWig",
        "H3K4me3": "https://www.encodeproject.org/files/ENCFF649DWF/@@download/ENCFF649DWF.bigWig",
        "ATAC": "https://www.encodeproject.org/files/ENCFF070LWM/@@download/ENCFF070LWM.bigWig",
    },
    "HeLa": {
        "DNase": "https://www.encodeproject.org/files/ENCFF159LZO/@@download/ENCFF159LZO.bigWig",
        "H3K4me3": "https://www.encodeproject.org/files/ENCFF447OHM/@@download/ENCFF447OHM.bigWig",
        "H3K27ac": "https://www.encodeproject.org/files/ENCFF816AHJ/@@download/ENCFF816AHJ.bigWig",
    },
}


def download_file(url: str, dest: str, overwrite: bool = False) -> bool:
    """Download a file from URL to local path.
    
    Returns True if downloaded, False if skipped (exists).
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        logger.info(f"Already exists: {dest}")
        return False
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading: {url}")
    logger.info(f"  → {dest}")
    
    try:
        urllib.request.urlretrieve(url, str(dest))
        logger.info(f"  Done ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return False


def clone_repo(url: str, dest: str) -> bool:
    """Clone a git repository."""
    dest = Path(dest)
    if dest.exists():
        logger.info(f"Already cloned: {dest}")
        return False
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cloning: {url}")
    
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True, capture_output=True, text=True,
        )
        logger.info(f"  Done: {dest}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  FAILED: {e.stderr}")
        return False


# ═══════════════════════════════════════════════════════════════
# Dataset-specific downloaders
# ═══════════════════════════════════════════════════════════════

def download_deephf(cfg: DictConfig, raw_dir: Path) -> None:
    """Download DeepHF dataset from GitHub.
    
    DeepHF contains ~60k sgRNA measurements across:
        - 3 cell lines: HEK293T, HCT116, HeLa
        - 3 Cas9 variants: WT-SpCas9, eSpCas9(1.1), SpCas9-HF1
    
    Source: https://github.com/Yichuan-Guo/DeepHF
    """
    dest = raw_dir / "deephf"
    
    # Clone full DeepHF repository
    clone_repo("https://github.com/Yichuan-Guo/DeepHF.git", str(dest))
    
    # Verify expected files
    expected_files = [
        "data/Doench_data.csv",
        "data/Wang_data.csv",
    ]
    for f in expected_files:
        p = dest / f
        if p.exists():
            logger.info(f"  ✓ Found: {f}")
        else:
            logger.warning(f"  ✗ Missing: {f}")


def download_crispron(cfg: DictConfig, raw_dir: Path) -> None:
    """Download CRISPRon dataset.
    
    Source: Kim et al. (2020), available from RTH Denmark.
    23,902 gRNAs with measured activities.
    """
    dest = raw_dir / "crispron"
    dest.mkdir(parents=True, exist_ok=True)
    
    # CRISPRon data files
    urls = [
        ("https://rth.dk/resources/crispr/crispron/data/CRISPRon_data.csv", "CRISPRon_data.csv"),
    ]
    
    for url, filename in urls:
        download_file(url, str(dest / filename))


def download_crispr_fmc(cfg: DictConfig, raw_dir: Path) -> None:
    """Download CRISPR-FMC benchmark datasets from GitHub.
    
    9 datasets: WT, ESP, HF, xCas9, SpCas9-NG, Sniper, HCT116, HeLa, HL60
    Source: https://github.com/xx0220/CRISPR-FMC
    """
    dest = raw_dir / "crispr_fmc"
    clone_repo("https://github.com/xx0220/CRISPR-FMC.git", str(dest))


def download_encode_tracks(cfg: DictConfig, raw_dir: Path) -> None:
    """Download ENCODE epigenomic bigWig tracks.
    
    For each cell line, downloads:
        - DNase-seq / ATAC-seq accessibility
        - H3K4me3 histone modification
        - H3K27ac histone modification
    
    Files are fold-change-over-control signal p-value bigWig (hg38).
    """
    dest = raw_dir / "encode"
    
    for cell_line, tracks in ENCODE_BIGWIG_URLS.items():
        for track_name, url in tracks.items():
            filename = f"{cell_line}_{track_name}.bigWig"
            download_file(url, str(dest / filename))


def download_offtarget_data(cfg: DictConfig, raw_dir: Path) -> None:
    """Download off-target datasets.
    
    - GUIDE-seq: ~150k validated off-target sites
    - CIRCLE-seq: ~50k off-target sites
    - CCLMoff processed data
    """
    dest = raw_dir / "offtarget"
    
    # GUIDE-seq
    clone_repo(
        "https://github.com/tsailabSJ/guideseq.git",
        str(dest / "guideseq"),
    )
    
    # CIRCLE-seq
    clone_repo(
        "https://github.com/tsailabSJ/circleseq.git",
        str(dest / "circleseq"),
    )
    
    # CCLMoff reference
    clone_repo(
        "https://github.com/duwa2/CCLMoff.git",
        str(dest / "CCLMoff"),
    )


# ═══════════════════════════════════════════════════════════════
# Main download function
# ═══════════════════════════════════════════════════════════════

def download_all(cfg: DictConfig) -> None:
    """Download all datasets specified in config.
    
    Usage:
        chromaguide data --stage download
    """
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ChromaGuide Data Acquisition")
    logger.info("=" * 60)
    
    logger.info("\n[1/5] DeepHF dataset...")
    download_deephf(cfg, raw_dir)
    
    logger.info("\n[2/5] CRISPRon dataset...")
    download_crispron(cfg, raw_dir)
    
    logger.info("\n[3/5] CRISPR-FMC benchmark...")
    download_crispr_fmc(cfg, raw_dir)
    
    logger.info("\n[4/5] ENCODE epigenomic tracks...")
    download_encode_tracks(cfg, raw_dir)
    
    logger.info("\n[5/5] Off-target datasets...")
    download_offtarget_data(cfg, raw_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Data acquisition complete!")
    logger.info("=" * 60)
