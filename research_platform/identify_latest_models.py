import json
import os
import re
from datetime import datetime

# Paths
INPUT_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/datasets/GLOBAL_FULL_INVENTORY.json"
OUTPUT_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/datasets/LATEST_VERSION_RECOMMENDATIONS.json"

# Hardware constraint
MAX_RAM_GB = 96

# BLEEDING EDGE MAPPING (2025 Context)
MAJOR_FAM_MAP = {
    "llama": ["llama-1", "llama-2", "llama-3", "llama-3.1", "llama-3.2", "llama-3.3", "llama-4"],
    "qwen": ["qwen", "qwen1.5", "qwen2", "qwen2.5", "qwen3", "qwen4"],
    "deepseek": ["deepseek-llm", "deepseek-v2", "deepseek-v2.5", "deepseek-v3", "deepseek-r1", "deepseek-r2"],
    "mistral": ["mistral-v0.1", "mistral-v0.2", "mistral-v0.3", "mistral-small", "mistral-large", "mistral-large-2", "mistral-large-3"],
    "gemma": ["gemma-1", "gemma-1.1", "gemma-2", "gemma-3"],
    "internvl": ["internvl-1", "internvl-2", "internvl-2.5", "internvl-3", "internvl-3.5", "internvl-4"],
    "granite": ["granite-1", "granite-2", "granite-3", "granite-3.1", "granite-4.0"],
    "nanbeige": ["nanbeige-1", "nanbeige-2", "nanbeige-3", "nanbeige-4"],
    "nemotron": ["nemotron-2", "nemotron-3", "nemotron-4"],
    "phi": ["phi-1", "phi-2", "phi-3", "phi-3.5", "phi-4"],
    "command": ["command-r", "command-r7b", "command-r-v1", "command-r-v2"],
    "flux": ["flux.1", "flux.2"],
    "stable-diffusion": ["sd-v1", "sdxl", "sd-3", "sd-3.5"],
    "internlm": ["internlm2", "internlm3", "intern-s1"]
}

OFFICIAL_ORGS = [
    "meta-llama", "Qwen", "deepseek-ai", "google", "mistralai", "microsoft",
    "stabilityai", "black-forest-labs", "01-ai", "nvidia", "apple", "ibm-granite",
    "baidu", "upstage", "internlm", "allenai", "openbmb", "togethercomputer", "internvl", "unsloth"
]

IGNORE_VERSION_NUMS = {224, 336, 384, 448, 512, 768, 1024, 1280, 14, 16, 32, 64, 70, 72, 80}

def detect_hierarchical_version(mid):
    # Normalized ID for matching
    mid_n = mid.lower().replace("_", ".").replace("-", ".")

    # 1. Check GEN_MAP for Architectural Generations
    for family, gens in MAJOR_FAM_MAP.items():
        if family in mid_n:
            for idx, g in enumerate(reversed(gens)):
                # Match g using the same normalization
                g_n = g.lower().replace("_", ".").replace("-", ".")
                # Also try matching without separators (e.g. Nanbeige4 vs nanbeige-4)
                g_solid = g_n.replace(".", "")
                mid_solid = mid_n.replace(".", "")

                if g_n in mid_n or g_solid in mid_solid:
                    gen_idx = len(gens) - 1 - idx
                    return (family, float(gen_idx) + 1000000.0)

    # 2. Extract numerical version
    # Replace _ with . for parsing version numbers like InternVL3_5
    clean = mid.replace("_", ".")
    clean = re.sub(r"(\d+(\.\d+)?)[bB](\W|$)", " ", clean)
    clean = re.sub(r"\d+bit", " ", clean)

    nums = re.findall(r"v?(\d+\.\d+|\d+)", clean)
    valid_nums = []
    for n in nums:
        try:
            v = float(n)
            if v in IGNORE_VERSION_NUMS: continue
            # Avoid picking up 2511 as a version if it looks like a date
            if v > 100 and v < 2050: continue
            valid_nums.append(v)
        except: continue

    if valid_nums:
        return ("Unknown", max(valid_nums))

    return ("Unknown", 0.0)

def get_group(mid):
    mid_short = mid.split('/')[-1].lower()
    for fam in MAJOR_FAM_MAP:
        if fam in mid_short: return fam
    return mid_short.split('-')[0].capitalize()

def analyze():
    print("Loading data...")
    with open(INPUT_FILE, 'r') as f: models = json.load(f)

    groups = {}
    print("Analyzing families...")
    for m in models:
        fam, ver = detect_hierarchical_version(m['id'])
        m['effective_family'] = fam if fam != "Unknown" else get_group(m['id'])
        m['parsed_version'] = ver
        m['parsed_date'] = parse_date(m.get('last_updated'))

        if m['effective_family'] not in groups: groups[m['effective_family']] = []
        groups[m['effective_family']].append(m)

    recs = {"reasoning_models": [], "multimodal_models": [], "code_models": [], "general_models": [], "image_generation_models": [], "embedding_models": []}

    print(f"Selecting best fits from families...")
    for fam_name, members in groups.items():
        members.sort(key=lambda x: (
            x['parsed_version'],
            x['id'].split('/')[0].lower() in [o.lower() for o in OFFICIAL_ORGS],
            x['parsed_date'],
            x.get('downloads', 0)
        ), reverse=True)

        max_v = members[0]['parsed_version']
        pool = [m for m in members if m['parsed_version'] == max_v]

        # Prioritize A3B/MoE
        moe_a3b = [m for m in pool if any(x in m['id'].lower() for x in ["a3b", "-moe-", "_moe_"])]
        if moe_a3b:
            pool = moe_a3b + [m for m in pool if m not in moe_a3b]

        best = None
        for m in pool:
            size_str = str(m.get('disk_gb', '0')).replace('GB', '').strip()
            try: size = float(size_str) if size_str != 'Unknown' else 0.0
            except: size = 0.0
            if size == 0: size = m.get('param_val', 0) * 0.7

            if size <= MAX_RAM_GB:
                if best is None or m.get('param_val', 0) > best.get('param_val', 0):
                    best = m
                    best['actual_size'] = size

        if best:
            human_ver = str(max_v)
            if max_v >= 1000000.0:
                lookup_fam = fam_name.lower()
                if lookup_fam in MAJOR_FAM_MAP:
                    idx = int(max_v - 1000000.0)
                    if idx < len(MAJOR_FAM_MAP[lookup_fam]):
                        human_ver = MAJOR_FAM_MAP[lookup_fam][idx]

            cat = categorize_model(best)
            entry = {
                "model_id": best['id'],
                "family": fam_name,
                "version": human_ver,
                "version_source": "Hierarchical version detection (Architectural Priority)",
                "size_estimate_gb": round(best.get('actual_size', 0), 1),
                "downloads": best.get('downloads', 0),
                "created_at": best.get('created_at', 'Unknown'),
                "last_updated": best.get('last_updated', 'Unknown'),
                "why_latest": f"Architectural generation {human_ver} is the absolute latest version found for {fam_name}.",
                "suitable_for": best.get('device_suitability', "General Inference")
            }
            if cat in recs: recs[cat].append(entry)

    for cat in recs: recs[cat] = sorted(recs[cat], key=lambda x: x['downloads'], reverse=True)
    with open(OUTPUT_FILE, 'w') as f: json.dump(recs, f, indent=2)
    print("DONE. Bleeding-edge update complete.")

def categorize_model(m):
    tags = [t.lower() for t in m.get('tags', [])]
    mid = m['id'].lower()
    if any(x in tags for x in ["image-generation", "flux"]) or "flux" in mid: return "image_generation_models"
    if any(x in tags for x in ["reasoning", "thinking"]) or any(x in mid for x in ["thinking", "r1", "qwq-", "reasoning", "scout"]): return "reasoning_models"
    if any(x in tags for x in ["multimodal", "vision", "vl"]) or any(x in mid for x in ["-vl", "-vision", "vlm", "internvl"]): return "multimodal_models"
    if any(x in tags for x in ["code", "coder"]) or "coder" in mid: return "code_models"
    if any(x in tags for x in ["embedding"]) or "embedding" in mid: return "embedding_models"
    return "general_models"

def parse_date(d):
    try: return datetime.strptime(d.split('+')[0].strip(), "%Y-%m-%d %H:%M:%S")
    except: return datetime.min

if __name__ == "__main__": analyze()
