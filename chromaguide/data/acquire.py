"""Data acquisition: download all required datasets.

Downloads real CRISPR datasets from verified public sources:
    1. CRISPR-FMC 9 benchmark datasets (WT, ESP, HF, xCas9, SpCas9-NG, Sniper,
       HCT116, HELA, HL60) — from xx0220/CRISPR-FMC GitHub repo
    2. DeepHF full dataset (~60K sgRNAs, 3 Cas9 variants) — from Rafid013/CRISPRpredSEQ
    3. ENCODE epigenomic tracks (bigWig files for HEK293T, HCT116, HeLa)
    4. Off-target datasets (GUIDE-seq, CIRCLE-seq)

Data sources (verified 2026-02):
    - CRISPR-FMC: Xiang et al. (2025), Frontiers in Genome Editing
      Datasets: Wang et al. 2019 (WT/ESP/HF), Kim et al. 2020 (xCas9/SpCas9-NG/Sniper),
                Hart et al. 2015 / Chuai et al. 2018 (HCT116/HELA/HL60)
    - DeepHF: Wang et al. (2019), Nature Communications 10:4284
    - CRISPRon: Xiang et al. (2021), Nature Communications 12:3238
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
# Verified Dataset URLs
# ═══════════════════════════════════════════════════════════════

# 9 CRISPR-FMC benchmark datasets (CSV: sgRNA, indel)
CRISPR_FMC_BASE_URL = "https://raw.githubusercontent.com/xx0220/CRISPR-FMC/main/datasets"
CRISPR_FMC_DATASETS = {
    # Large-scale (Wang et al. 2019): WT-SpCas9, eSpCas9(1.1), SpCas9-HF1
    "WT": {"url": f"{CRISPR_FMC_BASE_URL}/WT.csv", "n_expected": 55603,
           "cell_line": "HEK293T", "cas9_variant": "WT", "source": "Wang2019"},
    "ESP": {"url": f"{CRISPR_FMC_BASE_URL}/ESP.csv", "n_expected": 58616,
            "cell_line": "HEK293T", "cas9_variant": "eSpCas9", "source": "Wang2019"},
    "HF": {"url": f"{CRISPR_FMC_BASE_URL}/HF.csv", "n_expected": 56887,
            "cell_line": "HEK293T", "cas9_variant": "SpCas9-HF1", "source": "Wang2019"},
    # Medium-scale (Kim et al. 2020)
    "xCas9": {"url": f"{CRISPR_FMC_BASE_URL}/xCas.csv", "n_expected": 37738,
              "cell_line": "HEK293T", "cas9_variant": "xCas9", "source": "Kim2020"},
    "SpCas9_NG": {"url": f"{CRISPR_FMC_BASE_URL}/SpCas9-NG.csv", "n_expected": 30585,
                  "cell_line": "HEK293T", "cas9_variant": "SpCas9-NG", "source": "Kim2020"},
    "Sniper": {"url": f"{CRISPR_FMC_BASE_URL}/Sniper-Cas9.csv", "n_expected": 37794,
               "cell_line": "HEK293T", "cas9_variant": "Sniper-Cas9", "source": "Kim2020"},
    # Small-scale (Hart 2015 / Chuai 2018)
    "HCT116": {"url": f"{CRISPR_FMC_BASE_URL}/HCT116.csv", "n_expected": 4239,
               "cell_line": "HCT116", "cas9_variant": "WT", "source": "Hart2015"},
    "HELA": {"url": f"{CRISPR_FMC_BASE_URL}/HELA.csv", "n_expected": 8101,
             "cell_line": "HeLa", "cas9_variant": "WT", "source": "Hart2015"},
    "HL60": {"url": f"{CRISPR_FMC_BASE_URL}/HL60.csv", "n_expected": 2076,
             "cell_line": "HL60", "cas9_variant": "WT", "source": "Wang2014"},
}

# DeepHF full dataset (xlsx with all 3 variants in one file)
DEEPHF_URL = "https://raw.githubusercontent.com/Rafid013/CRISPRpredSEQ/master/deephf_data.xlsx"

# ENCODE epigenomic track URLs (bigWig, hg38, fold-change-over-control)
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

def download_crispr_fmc_datasets(raw_dir: Path) -> dict[str, Path]:
    """Download all 9 CRISPR-FMC benchmark datasets.

    Returns dict mapping dataset name → downloaded file path.
    """
    dest = raw_dir / "crispr_fmc" / "datasets"
    dest.mkdir(parents=True, exist_ok=True)
    downloaded = {}

    for name, info in CRISPR_FMC_DATASETS.items():
        filename = f"{name}.csv"
        filepath = dest / filename
        download_file(info["url"], str(filepath))

        if filepath.exists():
            # Verify row count
            with open(filepath) as f:
                n_lines = sum(1 for _ in f) - 1  # subtract header
            if n_lines != info["n_expected"]:
                logger.warning(
                    f"  ⚠ {name}: expected {info['n_expected']} rows, got {n_lines}"
                )
            else:
                logger.info(f"  ✓ {name}: {n_lines} rows verified")
            downloaded[name] = filepath

    return downloaded


def download_deephf_full(raw_dir: Path) -> Path | None:
    """Download the full DeepHF dataset (xlsx with 3 Cas9 variants).

    Source: Wang et al. (2019), ~60K sgRNAs across WT, eSpCas9, SpCas9-HF1.
    """
    dest = raw_dir / "deephf" / "deephf_data.xlsx"
    if download_file(DEEPHF_URL, str(dest)):
        return dest
    return dest if dest.exists() else None


def download_encode_tracks(raw_dir: Path) -> None:
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


def download_offtarget_data(raw_dir: Path) -> None:
    """Download off-target datasets.

    - GUIDE-seq: ~150k validated off-target sites
    - CIRCLE-seq: ~50k off-target sites
    """
    dest = raw_dir / "offtarget"

    clone_repo(
        "https://github.com/tsailabSJ/guideseq.git",
        str(dest / "guideseq"),
    )
    clone_repo(
        "https://github.com/tsailabSJ/circleseq.git",
        str(dest / "circleseq"),
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

    logger.info("\n[1/4] CRISPR-FMC benchmark datasets (9 datasets)...")
    downloaded = download_crispr_fmc_datasets(raw_dir)
    logger.info(f"  Downloaded {len(downloaded)}/9 datasets")

    logger.info("\n[2/4] DeepHF full dataset (supplementary)...")
    deephf_path = download_deephf_full(raw_dir)
    if deephf_path:
        logger.info(f"  ✓ DeepHF: {deephf_path}")

    logger.info("\n[3/4] ENCODE epigenomic tracks...")
    download_encode_tracks(raw_dir)

    logger.info("\n[4/4] Off-target datasets...")
    download_offtarget_data(raw_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Data acquisition complete!")
    logger.info("=" * 60)
