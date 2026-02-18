import requests
import json
import time
import urllib3
import re
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MANIFEST_FILES = [
    "/Users/studio/Desktop/PhD/Proposal/research_platform/GLOBAL_MODEL_MANIFEST.md",
    "/Users/studio/Desktop/PhD/Proposal/research_platform/SPECIALIZED_MANIFEST.md"
]

CACHE_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/METADATA_CACHE.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def fetch_model_details(model_id):
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        resp = requests.get(url, verify=False, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching details for {model_id}: {e}")
    return None

def fetch_storage_size(model_id):
    url = f"https://huggingface.co/api/models/{model_id}/tree/main"
    try:
        resp = requests.get(url, verify=False, timeout=10)
        if resp.status_code == 200:
            files = resp.json()
            total = 0
            for f in files:
                fname = f.get('path', '').lower()
                if fname.endswith(('.safetensors', '.bin', '.pt', '.pth', '.gguf', '.onnx', '.model')):
                    total += f.get('size', 0)
            return total
    except Exception as e:
        print(f"Error fetching size for {model_id}: {e}")
    return 0

def extract_metadata(model_id, data):
    if not data:
        return {"id": model_id, "params": "Unknown", "arch": "Unknown", "version": "Unknown"}

    tags = data.get('tags', [])
    config = data.get('config', {})

    # 1. Architecture
    arch = config.get('model_type', "Unknown")
    if arch == "Unknown":
        # Guess from tags
        for t in tags:
            if t in ["llama", "qwen2", "mistral", "phi3", "gemma2", "bert", "gpt2"]:
                arch = t
                break

    # 2. Parameters & Storage
    params = "Unknown"
    disk_size = 0.0

    # Calculate storage size from siblings (files)
    siblings = data.get('siblings', [])
    for s in siblings:
        rfilename = s.get('rfilename', '').lower()
        if rfilename.endswith(('.safetensors', '.bin', '.pt', '.pth', '.gguf', '.onnx')):
            # The API doesn't always provide size in siblings list without a specific param?
            # Actually, RepoInfo in API usually has size.
            pass

    # However, 'safetensors' key in API response often has 'total' (params) and 'total_size' (bytes)
    # Let's use that if available.
    safetensors = data.get('safetensors', {})
    if safetensors:
        if 'total' in safetensors:
            total = safetensors['total']
            if total > 1e9: params = f"{round(total/1e9, 1)}B"
            elif total > 1e6: params = f"{round(total/1e6, 1)}M"

        if 'total_size' in safetensors:
            disk_size = safetensors['total_size'] / (1024**3) # GB

    if disk_size == 0 and 'siblings' in data:
        # Fallback: some models have size in siblings
        for s in siblings:
            disk_size += s.get('size', 0) / (1024**3)

    if params == "Unknown":
        # Heuristic from name
        name = model_id.split('/')[-1]
        matches = re.findall(r"(\d+(\.\d+)?)[\s\-]?[bB]", name)
        if matches:
            params = f"{matches[-1][0]}B"

    # 3. Version
    version = "N/A"
    # Capture more versions like 2.5, 3.2, etc.
    version_match = re.search(r"[vV](\d+(\.\d+)*)|(\d+\.\d+)", model_id)
    if version_match:
        version = version_match.group(0)
    else:
        for t in tags:
            if re.search(r"\d+\.\d+", t):
                version = t
                break

    return {
        "id": model_id,
        "params": params,
        "disk_gb": f"{round(disk_size, 2)} GB" if disk_size > 0 else "Unknown",
        "arch": arch,
        "version": version
    }

def main():
    cache = load_cache()
    all_model_ids = set()

    # Step 1: Extract all model IDs from raw JSON audits for total coverage
    audit_files = [
        "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/MASTER_AUDIT_76.json",
        "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/SPECIALIZED_AUDIT.json"
    ]

    for af in audit_files:
        if not os.path.exists(af): continue
        with open(af, 'r') as f:
            data = json.load(f)
            # data is {org: [models]}
            for org, models in data.items():
                for m in models:
                    mid = m.get('name') or m.get('modelId')
                    if mid:
                        all_model_ids.add(mid)

    print(f"Found {len(all_model_ids)} total models from raw audits.")

    # Step 2: Fetch details for missing ones
    count = 0
    for mid in all_model_ids:
        if mid not in cache or "disk_gb" not in cache[mid] or cache[mid]["disk_gb"] == "Unknown":
            print(f"[{count}/{len(all_model_ids)}] Fetching {mid}...")
            details = fetch_model_details(mid)
            storage_bytes = fetch_storage_size(mid)

            meta = extract_metadata(mid, details)
            if storage_bytes > 0:
                meta["disk_gb"] = f"{round(storage_bytes/(1024**3), 2)} GB"

            cache[mid] = meta
            count += 1
            if count % 5 == 0:
                save_cache(cache)
            time.sleep(0.5)

    save_cache(cache)
    print("Enrichment complete. Cache saved.")

if __name__ == "__main__":
    main()
