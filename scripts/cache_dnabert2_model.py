#!/usr/bin/env python3
"""
Cache DNABERT-2 model to local HuggingFace cache directory
This should be run on the login node (not compute nodes) since compute nodes have no internet access

Usage:
    python3 cache_dnabert2_model.py
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

def cache_dnabert2():
    """Download and cache DNABERT-2 model locally"""
    
    print("=" * 80)
    print("DNABERT-2 MODEL CACHING")
    print("=" * 80)
    
    # Ensure cache directory exists and is writable
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Cache directory: {cache_dir}")
    print(f"Cache directory writable: {os.access(cache_dir, os.W_OK)}")
    
    # Set HuggingFace cache to this directory
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir.parent)
    
    model_name = "zhihan1996/DNABERT-2-117M"
    
    print(f"\nDownloading tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer cached successfully")
    except Exception as e:
        print(f"✗ Error caching tokenizer: {e}")
        sys.exit(1)
    
    print(f"\nDownloading model: {model_name}...")
    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_hidden_states=True
        )
        print(f"✓ Model cached successfully")
        print(f"  - Model config: hidden_size={model.config.hidden_size}")
        print(f"  - Model size: ~{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")
    except Exception as e:
        print(f"✗ Error caching model: {e}")
        sys.exit(1)
    
    # Verify cache contents
    cached_files = list(cache_dir.glob("**/config.json"))
    print(f"\nCached files found: {len(cached_files)}")
    for f in cached_files:
        print(f"  - {f}")
    
    print("\n✓ DNABERT-2 model successfully cached!")
    print(f"Cache location: {os.environ['HF_HOME']}")
    print("\nThis cache is now available for compute nodes to use offline.")
    
if __name__ == "__main__":
    cache_dnabert2()
