import re
from huggingface_hub import HfApi

# The Giants that need Small Siblings
GIANT_FAMILIES = {
    "DeepSeek": "deepseek-ai",
    "FLUX": "black-forest-labs",
    "InternVL": "OpenGVLab",
    "Molmo": "allenai",
    "Qwen": "Qwen", # Double check for 7B/8B max ver
    "Lumina": "Alpha-VLLM"
}

KEYWORDS_SMALL = ["7b", "8b", "6b", "4b", "3b", "1b", "schnell", "distill", "turbo", "lite", "small"]

def parse_size_gb(api, model_id):
    try:
        info = api.model_info(model_id, files_metadata=True)
        total_bytes = 0
        if info.siblings:
            for file in info.siblings:
                fname = file.rfilename.lower()
                if fname.endswith(('.safetensors', '.bin', '.pt', '.gguf', '.msgpack')):
                    if hasattr(file, 'size') and file.size:
                        total_bytes += file.size
        return total_bytes / (1024**3)
    except:
        return 999.0 # Assume huge if error

def extract_version(model_id):
    # Same version logic as before
    id_clean = re.sub(r'\d+(b|m|k)', '', model_id.lower())
    matches = re.findall(r'(\d+(?:\.\d+)?)', id_clean)
    valid = [float(x) for x in matches if float(x) < 20.0]
    return max(valid) if valid else 0.0

def find_small_sibling(api, family, org):
    print(f"   🔍 Hunting Small Sibling for {family}...")
    try:
        # Fetch verified org models
        models = api.list_models(author=org, sort="createdAt", direction="-1", limit=100, full=True)

        candidates = []
        for m in models:
            mid = m.id.lower()
            # Must match family name
            if family.lower() not in mid:
                continue

            # Must check size constraints logic via text first (fast)
            # Accept if it contains specific small keywords
            is_small_text = any(k in mid for k in KEYWORDS_SMALL)
            # Reject if it contains specific huge keywords
            is_huge = any(k in mid for k in ["70b", "30b", "235b", "pro", "large"])

            if is_small_text and not is_huge:
                # Double check accurate size
                # Heuristic: 8B usually fits.
                ver = extract_version(m.id)
                candidates.append((ver, m))

        # Sort by Version DESC
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Verify the top candidate size
        for ver, m in candidates:
            gb = parse_size_gb(api, m.id)
            if gb < 20.0: # strict <20GB (fits <10B params usually)
                print(f"      ✅ FOUND: {m.id} (v{ver}, {gb:.2f} GB)")
                return m.id, ver, gb
            else:
                print(f"      ❌ REJECTED: {m.id} (Too big: {gb:.2f} GB)")

    except Exception as e:
        print(f"Error: {e}")

    return None, 0.0, 0.0

def run_hunt():
    api = HfApi()
    print("🚀 STARTING SIBLING HUNT (<10B Verified)...")

    for fam, org in GIANT_FAMILIES.items():
        find_small_sibling(api, fam, org)

if __name__ == "__main__":
    run_hunt()
