import json
import os
import re
import requests
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Paths
AUDIT_76_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/MASTER_AUDIT_76.json"
SPECIALIZED_AUDIT_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/SPECIALIZED_AUDIT.json"
METADATA_CACHE_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/METADATA_CACHE.json"

OUT_GLOBAL = "/Users/studio/Desktop/PhD/Proposal/research_platform/GLOBAL_MODEL_MANIFEST.md"
OUT_SPECIALIZED = "/Users/studio/Desktop/PhD/Proposal/research_platform/SPECIALIZED_MANIFEST.md"

GEN_MAP = {
    "llama": ["llama-1", "llama-2", "llama-3", "llama-3.1", "llama-3.2", "llama-3.3"],
    "gemma": ["gemma-1", "gemma-1.1", "gemma-2", "gemma-3"],
    "qwen": ["qwen", "qwen1.5", "qwen2", "qwen2.5", "qwen3"],
    "mistral": ["mistral-v0.1", "mistral-v0.2", "mistral-v0.3", "mistral-small", "mistral-large", "mistral-large-2", "mistral-large-3"],
    "deepseek": ["deepseek-llm", "deepseek-v2", "deepseek-v3", "deepseek-r1"],
    "yi": ["yi-6b", "yi-1.5"],
    "internlm": ["internlm", "internlm2", "internlm2.5", "internlm3", "intern-s1"],
    "cohere": ["aya-101", "aya-23", "aya-expanse", "command-r", "command-r-plus", "command-r7b"],
    "bigcode": ["starcoder", "starcoder2"],
    "chatglm": ["chatglm-6b", "chatglm2", "chatglm3", "glm-4"],
    "baidu": ["ernie-3", "ernie-4", "ernie-4.5"]
}

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
            if t in ["llama", "qwen2", "mistral", "phi3", "gemma2", "bert"]: arch = t; break

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

def detect_generation(name):
    name_lower = name.lower()
    for family, gens in GEN_MAP.items():
        for idx, g in enumerate(reversed(gens)):
            if g in name_lower: return (family, len(gens) - 1 - idx)
    return (None, -1)

def filter_latest(models):
    max_gens = {}
    for m in models:
        fam, gen = detect_generation(m['name'])
        if fam: max_gens[fam] = max(max_gens.get(fam, -1), gen)
    return [m for m in models if not detect_generation(m['name'])[0] or detect_generation(m['name'])[1] >= max_gens[detect_generation(m['name'])[0]]]

def main():
    cache = {}
    if os.path.exists(METADATA_CACHE_FILE):
        with open(METADATA_CACHE_FILE, 'r') as f: cache = json.load(f)

    # Load Audits
    with open(AUDIT_76_FILE, 'r') as f: g_data = json.load(f)
    with open(SPECIALIZED_AUDIT_FILE, 'r') as f: s_data = json.load(f)

    # Flatten & Pre-Filter Global (Top 200 likely survivors)
    g_flat = []
    for org, models in g_data.items():
        for m in models: g_flat.append({"name": m['name'], "likes": m.get('likes', 0), "tags": m.get('tags', [])})
    g_filtered = filter_latest(g_flat)
    g_filtered.sort(key=lambda x: x['likes'], reverse=True)
    g_top = g_filtered[:150]

    # Flatten & Pre-Filter Specialized
    CATEGORIES = ["🛡️ Safeguard", "🛠️ Tool Use", "⚡ Edge/Mobile", "🧠 Specialized Instruct"]
    s_by_cat = {c: [] for c in CATEGORIES}
    unique_ids = set(m['name'] for m in g_top)

    for org, models in s_data.items():
        for m in models:
            for cat in m.get('categories', []):
                if cat in s_by_cat: s_by_cat[cat].append(m)

    for cat in CATEGORIES:
        s_by_cat[cat] = filter_latest(s_by_cat[cat])
        s_by_cat[cat].sort(key=lambda x: x.get('likes', 0), reverse=True)
        for m in s_by_cat[cat][:20]: unique_ids.add(m['name'])

    # ENRICH SURVIVORS
    print(f"Enriching {len(unique_ids)} survivors...")
    for idx, mid in enumerate(unique_ids):
        if mid not in cache or cache[mid].get("disk_gb") == "Unknown":
            print(f"[{idx}/{len(unique_ids)}] Fetching {mid}...")
            d = fetch_details(mid)
            sz = fetch_size(mid)
            meta = extract_meta(mid, d)
            if sz > 0: meta["disk_gb"] = f"{round(sz/(1024**3), 2)} GB"
            cache[mid] = meta
            if idx % 10 == 0:
                with open(METADATA_CACHE_FILE, 'w') as f: json.dump(cache, f, indent=2)
            time.sleep(0.5)

    with open(METADATA_CACHE_FILE, 'w') as f: json.dump(cache, f, indent=2)

    # RENDER SPECIALIZED
    spec_lines = ["# Specialized Capabilities (Latest Gen Enriched)", "> **Constraint**: Latest Generations with Disk Size, Architecture, and Version metadata.", ""]
    for cat in CATEGORIES:
        spec_lines.append(f"## {cat}\n| Model | Params | Disk | Arch | Version | Likes | Org | Tags |\n|---|---|---|---|---|---|---|---|")
        seen = set()
        for m in s_by_cat[cat][:20]:
            if m['name'] in seen: continue
            seen.add(m['name'])
            meta = cache.get(m['name'], {})
            oname = m['name'].split('/')[0]
            cname = m['name'].split('/')[-1]
            tstr = ", ".join(m.get('tags', [])[:3])
            spec_lines.append(f"| [{cname}](https://huggingface.co/{m['name']}) | {meta.get('params','Unknown')} | {meta.get('disk_gb','Unknown')} | {meta.get('arch','Unknown')} | {meta.get('version','N/A')} | {m['likes']} | {oname} | {tstr} |")
        spec_lines.append("")
    with open(OUT_SPECIALIZED, 'w') as f: f.write("\n".join(spec_lines))

    # RENDER GLOBAL
    glob_lines = ["# Global Model Manifest (Efficiency Focused)", "> **Constraint**: <= 32B Parameters, Latest Generations Only.", "", "| Organization | Repository | Params | Disk | Arch | Version | Likes | Tags |\n|---|---|---|---|---|---|---|---|"]
    for m in g_top:
        meta = cache.get(m['name'], {})
        oname = m['name'].split('/')[0]
        cname = m['name'].split('/')[-1]
        tstr = ", ".join(m.get('tags', [])[:3])
        glob_lines.append(f"| {oname} | [{cname}](https://huggingface.co/{m['name']}) | {meta.get('params','Unknown')} | {meta.get('disk_gb','Unknown')} | {meta.get('arch','Unknown')} | {meta.get('version','N/A')} | {m['likes']} | {tstr} |")
    with open(OUT_GLOBAL, 'w') as f: f.write("\n".join(glob_lines))
    print("All manifests generated.")

if __name__ == "__main__":
    main()
