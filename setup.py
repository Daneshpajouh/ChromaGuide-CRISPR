"""ChromaGuide: Multi-Modal CRISPR-Cas9 sgRNA Design Framework."""
from setuptools import setup, find_packages

setup(
    name="chromaguide",
    version="0.1.0",
    author="Amir Daneshpajouh",
    author_email="amir@mystorax.com",
    description="Chromatin-aware guide RNA design with conformal prediction",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "biopython>=1.81",
        "pyBigWig>=0.3.22",
        "transformers>=4.36.0",
        "einops>=0.7.0",
        "optuna>=3.4.0",
        "wandb>=0.16.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "rich>=13.7.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "chromaguide=chromaguide.cli:main",
        ],
    },
)
