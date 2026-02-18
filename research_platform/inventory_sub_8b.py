import json
import os
import re
import requests
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Paths
AUDIT_76_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/MASTER_AUDIT_76.json"
METADATA_CACHE_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/METADATA_CACHE.json"
OUT_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/SUB_8B_MODEL_INVENTORY.md"

def fetch_details(mid):
    url = f"https://huggingface.co/api/models/{mid}"
    try:
        r = requests.get(url, verify=False, timeout=10)
        return r.json() if r.status_code == 200 else None
    except: return None

def fetch_size(mid):
    url = f"https://huggingface.co/api/models/{mid}/tree/main"
    try:
        r = requests.get(url, verify=False, timeout=10)
        if r.status_code == 200:
            return sum(f.get('size', 0) for f in r.json() if f.get('path', '').lower().endswith(('.safetensors', '.bin', '.pt', '.pth', '.gguf', '.onnx', '.model')))
    except: pass
    return 0

def extract_meta(mid, data):
    if not data: return {"id": mid, "params": "Unknown", "disk_gb": "Unknown", "arch": "Unknown", "version": "N/A"}
    tags = data.get('tags', [])
    config = data.get('config', {})
    arch = config.get('model_type', "Unknown")
    if arch == "Unknown":
        for t in tags:
            if t in ["llama", "qwen2", "mistral", "phi3", "gemma2", "bert", "gpt2"]: arch = t; break

    params = "Unknown"
    st = data.get('safetensors', {})
    if st and 'total' in st:
        total = st['total']
        if total > 1e9: params = f"{round(total/1e9, 1)}B"
        elif total > 1e6: params = f"{round(total/1e6, 1)}M"

    version = "N/A"
    v_match = re.search(r"[vV](\d+(\.\d+)*)|(\d+\.\d+)", mid)
    if v_match: version = v_match.group(0)

    return {"id": mid, "params": params, "disk_gb": "Unknown", "arch": arch, "version": version}

def parse_size_heuristic(name):
    # Try to find B or M in name
    match = re.search(r"(\d+(\.\d+)?)[\s\-]?[bB]", name)
    if match: return float(match.group(1))
    match = re.search(r"(\d+(\.\d+)?)[\s\-]?[mM]", name)
    if match: return float(match.group(1)) / 1000.0
    return None

def main():
    if not os.path.exists(AUDIT_76_FILE):
        print("Error: MASTER_AUDIT_76.json not found.")
        return

    with open(AUDIT_76_FILE, 'r') as f:
        data = json.load(f)

    cache = {}
    if os.path.exists(METADATA_CACHE_FILE):
        with open(METADATA_CACHE_FILE, 'r') as f: cache = json.load(f)

    sub_8b_models = []

    print("Filtering models < 8B...")
    for org, models in data.items():
        for m in models:
            mid = m['name']
            size = parse_size_heuristic(mid)

            # If heuristic fails, check cache if we have it
            if size is None and mid in cache:
                param_str = cache[mid].get("params", "")
                if "B" in param_str:
                    try: size = float(param_str.replace("B", ""))
                    except: pass
                elif "M" in param_str:
                    try: size = float(param_str.replace("M", "")) / 1000.0
                    except: pass

            if size is not None and size < 8.1: # Allow slight buffer for decimals
                sub_8b_models.append({
                    "name": mid,
                    "likes": m.get('likes', 0),
                    "tags": m.get('tags', []),
                    "est_size": size
                })

    # Sort and pick top 200 by likes to keep it manageable and high quality
    sub_8b_models.sort(key=lambda x: x['likes'], reverse=True)
    targets = sub_8b_models[:200]

    print(f"Enriching top {len(targets)} sub-8B survivors...")
    for idx, m in enumerate(targets):
        mid = m['name']
        if mid not in cache or cache[mid].get("disk_gb") == "Unknown":
            print(f"[{idx}/{len(targets)}] Fetching {mid}...")
            d = fetch_details(mid)
            sz = fetch_size(mid)
            meta = extract_meta(mid, d)
            if sz > 0: meta["disk_gb"] = f"{round(sz/(1024**3), 2)} GB"
            cache[mid] = meta
            if idx % 10 == 0:
                with open(METADATA_CACHE_FILE, 'w') as f: json.dump(cache, f, indent=2)
            time.sleep(0.5)

    with open(METADATA_CACHE_FILE, 'w') as f: json.dump(cache, f, indent=2)

    # Categorize
    ultra_light = [] # < 1B
    edge = []        # 1B - 3B
    portable = []    # 3B - 8B

    for m in targets:
        mid = m['name']
        meta = cache.get(mid, {})
        param_val = 0.0
        param_str = meta.get("params", "")
        if "B" in param_str:
            try: param_val = float(param_str.replace("B", ""))
            except: param_val = m['est_size']
        elif "M" in param_str:
            try: param_val = float(param_str.replace("M", "")) / 1000.0
            except: param_val = m['est_size']
        else:
            param_val = m['est_size']

        if param_val < 1.0: ultra_light.append((m, meta))
        elif param_val < 3.5: edge.append((m, meta))
        else: portable.append((m, meta))

    # Render
    lines = ["# Sub-8B Model Inventory", "> **Focus**: Highly portable models ranging from sub-1B to 8B parameters.", ""]

    sections = [
        ("Ultra-Light (<1B Parameters)", ultra_light),
        ("Edge/Mobile (1B-3B Parameters)", edge),
        ("Portable/Standard (3B-8B Parameters)", portable)
    ]

    for title, model_list in sections:
        lines.append(f"## {title}")
        lines.append("| Model | Params | Disk | Arch | Version | Likes | Org | Tags |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m, meta in model_list:
            oname = m['name'].split('/')[0]
            cname = m['name'].split('/')[-1]
            tstr = ", ".join(m['tags'][:3])
            lines.append(f"| [{cname}](https://huggingface.co/{m['name']}) | {meta.get('params','Unknown')} | {meta.get('disk_gb','Unknown')} | {meta.get('arch','Unknown')} | {meta.get('version','N/A')} | {m['likes']} | {oname} | {tstr} |")
        lines.append("")

    with open(OUT_FILE, 'w') as f:
        f.write("\n".join(lines))
    print(f"Inventory generated at {OUT_FILE}")

if __name__ == "__main__":
    main()
