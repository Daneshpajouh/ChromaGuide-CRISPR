#!/usr/bin/env python3
"""Download and cache DNABERT-2 model locally (no custom code to avoid triton dep)"""
from transformers import AutoTokenizer, BertModel
import sys
import os

print("📥 Downloading DNABERT-2 model (standard BERT architecture)...")
sys.stderr.flush()

try:
    print("  → Loading tokenizer...", flush=True)
    t = AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M')
    print("  ✓ Tokenizer loaded", flush=True)

    print("  → Loading model...", flush=True)
    m = BertModel.from_pretrained('zhihan1996/DNABERT-2-117M')
    print("  ✓ Model loaded", flush=True)

    os.makedirs('models', exist_ok=True)
    print("  → Saving tokenizer...", flush=True)
    t.save_pretrained('models/dnabert2')
    print("  ✓ Tokenizer saved", flush=True)

    print("  → Saving model...", flush=True)
    m.save_pretrained('models/dnabert2')
    print("  ✓ Model saved", flush=True)

    print("✅ DNABERT-2 cached successfully in models/dnabert2")
    print(f"   Model size: {os.popen('du -sh models/dnabert2').read().strip()}")

except Exception as e:
    print(f"❌ Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
