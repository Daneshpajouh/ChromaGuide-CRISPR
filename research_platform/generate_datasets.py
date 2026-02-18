import json
import os
import re
import csv

# Paths
AUDIT_76_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/MASTER_AUDIT_76.json"
RAW_CACHE_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/RAW_METADATA_FULL.json"
OUT_DIR = "/Users/studio/Desktop/PhD/Proposal/research_platform/datasets"

if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

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

def detect_generation(name):
    name_lower = name.lower()
    for family, gens in GEN_MAP.items():
        for idx, g in enumerate(reversed(gens)):
            if g in name_lower: return (family, len(gens) - 1 - idx)
    return (None, -1)

def filter_latest(models):
    if not models: return []
    max_gens = {}
    for m in models:
        fam, gen = detect_generation(m.get('id', m.get('name', '')))
        if fam: max_gens[fam] = max(max_gens.get(fam, -1), gen)

    # Logic: Keep if (no family matched) OR (gen >= max_gen) OR (likes > 50) OR (downloads > 1000)
    # This prevents dropping highly popular models that might be labeled slightly differently
    final = []
    for m in models:
        fam, gen = detect_generation(m.get('id', m.get('name', '')))
        is_latest = not fam or gen >= max_gens.get(fam, -1)
        is_popular = m.get('likes', 0) > 50 or m.get('downloads', 0) > 1000
        if is_latest or is_popular:
            final.append(m)
    return final

def parse_param_val(p_str):
    if not p_str or p_str == "Unknown": return 0.0
    try:
        p_str = p_str.replace("~", "").strip()
        if "B" in p_str: return float(p_str.replace("B", ""))
        if "M" in p_str: return float(p_str.replace("M", "")) / 1000.0
    except: pass
    return 0.0

def get_suitability(p_val, arch):
    if p_val == 0: return "Various (Check Size)"
    if p_val < 1.0: return "Ultra-Light (Mobile/IoT)"
    if p_val < 4.0: return "Edge-Efficient (Phone/Tablet)"
    if p_val < 9.0: return "Portable (Laptop/Desktop)"
    if p_val < 35.0: return "Workstation (High-RAM)"
    return "Server-Grade"

def extract_meta(mid, data):
    meta = {"id": mid, "params": "Unknown", "disk_gb": "Unknown", "arch": "Unknown", "version": "N/A", "created_at": "Unknown", "last_updated": "Unknown"}
    if not data or not isinstance(data, dict): return meta
    try:
        meta["created_at"] = data.get("createdAt") or data.get("created_at") or "Unknown"
        meta["last_updated"] = data.get("lastModified") or data.get("last_modified") or "Unknown"

        # Comprehensive storage extraction
        storage_bytes = data.get('usedStorage') or data.get('modelSize') or 0
        if not storage_bytes:
            sibs = data.get('siblings', [])
            if isinstance(sibs, list):
                for s in sibs:
                    fname = s.get('rfilename') if isinstance(s, dict) else getattr(s, 'rfilename', "")
                    fsize = s.get('size', 0) if isinstance(s, dict) else getattr(s, 'size', 0)
                    if fname and fname.lower().endswith(('.safetensors', '.bin', '.pt', '.pth', '.gguf', '.onnx', '.model', '.pdparams', '.params', '.t7', '.h5')):
                        storage_bytes += fsize

        # Parameter extraction
        params = "Unknown"
        st = data.get('safetensors')
        if not st or (isinstance(st, dict) and 'total' not in st):
            st = (data.get('cardData') or {}).get('safetensors')

        if isinstance(st, dict) and st.get('total'):
            total = st['total']
            if total > 1e9: params = f"{round(total/1e9, 1)}B"
            elif total > 1e6: params = f"{round(total/1e6, 1)}M"
            if not storage_bytes and total > 0: storage_bytes = total * 2

        # Architecture and Tags
        tags = data.get('tags', [])
        config = data.get('config', {})
        arch = "Unknown"
        if isinstance(config, dict):
            arch = config.get('model_type', "Unknown")
        if arch == "Unknown":
            for t in tags:
                if t in ["llama", "qwen2", "mistral", "phi3", "gemma2", "bert", "gpt2", "ernie"]: arch = t; break
        meta["arch"] = arch

        # Fallback heuristics for params
        if params == "Unknown":
            p_match = re.search(r"(\W|^)(\d+(\.\d+)?[bmBM])(\W|$)", mid)
            if p_match:
                params = p_match.group(2).upper()
            elif storage_bytes > 0:
                gb = storage_bytes / (1024**3)
                if gb > 0.1:
                    est = round(gb / 2.1, 1)
                    params = f"~{est}B" if est >= 1.0 else f"~{int(est*1000)}M"

        meta["params"] = params
        if storage_bytes > 0:
            meta["disk_gb"] = f"{round(storage_bytes/(1024**3), 2)} GB"

        v_match = re.search(r"[vV](\d+(\.\d+)*)|(\d+\.\d+)", mid)
        if v_match: meta["version"] = v_match.group(0)
    except: pass
    return meta

def save_csv(data, filename):
    if not data: return
    all_keys = set()
    for row in data: all_keys.update(row.keys())
    # Fixed priority for ID
    sorted_keys = sorted(list(all_keys))
    if 'id' in sorted_keys:
        sorted_keys.remove('id')
        sorted_keys.insert(0, 'id')
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        dw = csv.DictWriter(f, fieldnames=sorted_keys)
        dw.writeheader()
        for row in data:
            rc = row.copy()
            for k, v in rc.items():
                if isinstance(v, (list, dict)): rc[k] = str(v)
                elif v is None: rc[k] = ""
            dw.writerow(rc)

def main():
    if os.path.exists(AUDIT_76_FILE):
        with open(AUDIT_76_FILE, 'r') as f: audit = json.load(f)
        orgs = list(audit.keys())
    else:
        orgs = []

    raw_cache = {}
    if os.path.exists(RAW_CACHE_FILE):
        with open(RAW_CACHE_FILE, 'r') as f:
            raw_cache = json.load(f)
            print(f"✅ Loaded {len(raw_cache)} models from cache.")

    all_models = []
    for mid, raw in raw_cache.items():
        if isinstance(raw, str):
            try: raw = json.loads(raw.replace("'", '"'))
            except: continue
        meta = extract_meta(mid, raw)
        m = {
            "id": mid,
            "name": mid,
            "likes": raw.get('likes', 0) if isinstance(raw, dict) else 0,
            "downloads": raw.get('downloads', 0) if isinstance(raw, dict) else 0,
            "tags": raw.get('tags', []) if isinstance(raw, dict) else [],
            **meta
        }
        m['url'] = f"https://huggingface.co/{mid}"
        m['param_val'] = parse_param_val(m.get('params'))
        m['device_suitability'] = get_suitability(m['param_val'], m.get('arch'))
        m['org'] = mid.split('/')[0] if '/' in mid else 'Unknown'
        all_models.append(m)

    target_orgs = set(orgs)

    # 1. GLOBAL FULL INVENTORY (Everything captured)
    full_inventory = sorted(all_models, key=lambda x: (x['likes'], x['downloads']), reverse=True)

    # 2. GLOBAL EFFICIENCY (<= 35B, Target Orgs, Latest Generation)
    g_pool = [m for m in all_models if m['org'] in target_orgs and m['param_val'] <= 35.0]
    g_eff = filter_latest(g_pool)

    # 3. SPECIALIZED INVENTORY (All Target Orgs, Latest Generation)
    s_pool = [m for m in all_models if m['org'] in target_orgs]
    s_inv = filter_latest(s_pool)

    # 4. SUB-8B EFFICIENT (<= 9B, Any Org from pool, Latest Generation)
    l_pool = [m for m in all_models if 0 < m['param_val'] <= 9.0]
    l_eff = filter_latest(l_pool)

    datasets = {
        "GLOBAL_FULL_INVENTORY": full_inventory,
        "GLOBAL_EFFICIENCY_DATASET": g_eff,
        "SPECIALIZED_INVENTORY": s_inv,
        "SUB_8B_EFFICIENT_DATASET": l_eff
    }

    for name, data in datasets.items():
        json_path = os.path.join(OUT_DIR, f"{name}.json")
        csv_path = os.path.join(OUT_DIR, f"{name}.csv")
        with open(json_path, 'w') as f: json.dump(data, f, indent=2, default=str)
        save_csv(data, csv_path)
        print(f"📦 Exported {name}: {len(data)} models")

    print(f"\nFinal Audit Summary:")
    print(f"Total Unique Models Scanned: {len(all_models)}")
    print(f"Total Dataset Entries: {sum(len(d) for d in datasets.values())}")

if __name__ == "__main__":
    main()
