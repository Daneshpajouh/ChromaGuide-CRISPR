import re
from huggingface_hub import HfApi

# The Target: LATEST Version Names + Small Modifiers
# We want "DeepSeek-V3" not "DeepSeek-Coder-1"
TARGETS = {
    "DeepSeek": ["DeepSeek-V3", "DeepSeek-V3.2", "DeepSeek-V3-Distill", "DeepSeek-R1-Distill"],
    "FLUX": ["FLUX.2", "FLUX-2", "FLUX.2-schnell"],
    "InternVL": ["InternVL3", "InternVL-3", "InternVL3.5"],
    "Qwen": ["Qwen3", "Qwen-3"],
    "Molmo": ["Molmo2", "Molmo-2"],
    "Lumina": ["Lumina-Image-3", "Lumina-3"]
}

MODIFIERS = ["distill", "lite", "small", "mini", "micro", "nano", "quant", "gguf", "mlx", "awq", "gptq", "4bit", "q4", "8b", "7b", "4b", "3b"]

def parse_size_gb(api, model_id):
    try:
        info = api.model_info(model_id, files_metadata=True)
        total = 0
        if info.siblings:
            for f in info.siblings:
                if f.rfilename.endswith(('.safetensors', '.bin', '.gguf')):
                    total += f.size
        return total / (1024**3)
    except: return 999.0

def run_variant_hunt():
    api = HfApi()
    print("🚀 HUNTING SPECIFIC LATEST-VERSION VARIANTS (<12B)...")

    for family, versions in TARGETS.items():
        print(f"\n📂 {family}: Searching for {versions}...")

        candidates = []
        for v_name in versions:
            # Search for "DeepSeek-V3"
            try:
                models = api.list_models(search=v_name, sort="createdAt", direction="-1", limit=50, full=True)
                for m in models:
                    mid = m.id.lower()
                    # Must contain the version string explicitly (case insensitive)
                    if v_name.lower() not in mid: continue

                    # Must NOT be huge
                    if any(x in mid for x in ["671b", "70b", "30b", "236b"]):
                        if "distill" not in mid and "quant" not in mid:
                            continue

                    # Check size
                    # First quick text check
                    if parse_size_gb(api, m.id) < 15.0: # 15GB tolerance for 8B params
                         candidates.append(m)
            except: pass

        # Deduplicate and Sort
        unique = {c.id: c for c in candidates}.values()
        sorted_c = sorted(unique, key=lambda x: x.created_at, reverse=True)

        found = False
        for c in sorted_c:
            gb = parse_size_gb(api, c.id)
            if gb < 15.0:
                print(f"   ✅ FOUND: {c.id}")
                print(f"      Size: {gb:.2f} GB | Released: {str(c.created_at)[:10]}")
                found = True
                break # Found the newest small one

        if not found:
            print(f"   ❌ No small {family} {versions} found. (Checked {len(unique)} candidates)")

if __name__ == "__main__":
    run_variant_hunt()
