import json
import os
import re
from datetime import datetime

# Paths
INPUT_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/datasets/GLOBAL_FULL_INVENTORY.json"
OUTPUT_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/datasets/SHORTLIST_EFFICIENT_LATEST.json"

# USER CONSTRAINTS
MAX_DENSE_PARAMS = 15.0  # ~14B
MAX_MOE_TOTAL_PARAMS = 35.0  # ~32B
FAVOR_ACTIVE_PARAMS = 3.0  # A3B
ULTRA_LIGHT_THRESHOLD = 2.0 # 2B or less

# ORDER MATTERS: Put specific descendants/variants before general base families
MAJOR_FAM_MAP = {
    "functiongemma": ["functiongemma"],
    "smollm": ["smollm", "smollm2"],
    "granite": ["granite-1", "granite-2", "granite-3", "granite-3.1", "granite-4.0"],
    "llama": ["llama-1", "llama-2", "llama-3", "llama-3.1", "llama-3.2", "llama-3.3", "llama-4"],
    "qwen": ["qwen", "qwen1.5", "qwen2", "qwen2.5", "qwen3", "qwen4"],
    "deepseek": ["deepseek-llm", "deepseek-v2", "deepseek-v2.5", "deepseek-v3", "deepseek-r1", "deepseek-r2"],
    "mistral": ["mistral-v0.1", "mistral-v0.2", "mistral-v0.3", "mistral-small", "mistral-large", "mistral-large-2", "mistral-large-3"],
    "gemma": ["gemma-1", "gemma-1.1", "gemma-2", "gemma-3"],
    "internvl": ["internvl-1", "internvl-2", "internvl-2.5", "internvl-3", "internvl-3.5", "internvl-4"],
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
    "baidu", "upstage", "internlm", "allenai", "openbmb", "togethercomputer", "internvl", "unsloth", "HuggingFaceTB"
]

IGNORE_VERSION_NUMS = {224, 336, 384, 448, 512, 768, 1024, 1280, 14, 16, 32, 64, 70, 72, 80}

def detect_hierarchical_version(mid):
    mid_n = mid.lower().replace("_", ".").replace("-", ".")
    for family, gens in MAJOR_FAM_MAP.items():
        # Use word boundaries or check for start/end of segment to avoid partial family match
        # e.g. "functiongemma" shouldn't match "gemma" if "functiongemma" is its own family
        if family in mid_n:
            # Check for the family name as a distinct part
            parts = mid_n.split('.')
            if family in parts or any(family in p for p in parts):
                for idx, g in enumerate(reversed(gens)):
                    g_n = g.lower().replace("_", ".").replace("-", ".")
                    g_solid = g_n.replace(".", "")
                    mid_solid = mid_n.replace(".", "")
                    if g_n in mid_n or g_solid in mid_solid:
                        return (family, float(len(gens) - 1 - idx) + 1000000.0)
                # If family matches but no specific generation matches, return family base version
                return (family, 1000000.0)

    clean = mid.replace("_", ".")
    clean = re.sub(r"(\d+(\.\d+)?)[bB](\W|$)", " ", clean)
    nums = re.findall(r"v?(\d+\.\d+|\d+)", clean)
    valid_nums = [float(n) for n in nums if float(n) not in IGNORE_VERSION_NUMS and float(n) < 100]
    return ("Unknown", max(valid_nums) if valid_nums else 0.0)

def categorize_model(m):
    tags = [t.lower() for t in m.get('tags', [])]
    mid = m['id'].lower()

    # Tool-Calling / Ultra-Lightweight
    if any(x in mid for x in ["function", "tool-use", "tool-calling"]) or "smollm" in mid or ("granite" in mid and any(x in mid for x in ["micro", "tiny", "nano"])):
        params = m.get('param_val', 0)
        if params > 0 and params <= ULTRA_LIGHT_THRESHOLD:
            return "tool_calling_ultra_light"

    if any(x in tags for x in ["image-generation", "flux"]) or "flux" in mid: return "image_generation_models"
    if any(x in tags for x in ["reasoning", "thinking"]) or any(x in mid for x in ["thinking", "r1", "qwq-", "reasoning", "scout"]): return "reasoning_models"
    if any(x in tags for x in ["multimodal", "vision", "vl"]) or any(x in mid for x in ["-vl", "-vision", "vlm", "internvl"]): return "multimodal_models"
    if any(x in tags for x in ["code", "coder"]) or "coder" in mid: return "code_models"
    if any(x in tags for x in ["embedding"]) or "embedding" in mid: return "embedding_models"
    return "general_models"

def parse_date(d):
    try: return datetime.strptime(d.split('+')[0].strip(), "%Y-%m-%d %H:%M:%S")
    except: return datetime.min

def analyze():
    print("Loading data...")
    with open(INPUT_FILE, 'r') as f: models = json.load(f)

    family_groups = {}
    max_versions = {}
    for m in models:
        fam, ver = detect_hierarchical_version(m['id'])
        m['effective_family'] = fam
        m['parsed_version'] = ver
        if fam not in family_groups: family_groups[fam] = []
        family_groups[fam].append(m)
        max_versions[fam] = max(max_versions.get(fam, 0), ver)

    shortlist = {
        "tool_calling_ultra_light": [],
        "reasoning_models": [],
        "multimodal_models": [],
        "code_models": [],
        "general_models": [],
        "image_generation_models": [],
        "embedding_models": []
    }

    print("Producing efficient shortlist...")
    for m in models:
        fam = m['effective_family']
        if m['parsed_version'] < max_versions[fam]: continue

        mid = m['id'].lower()
        params = m.get('param_val', 0)
        is_moe = any(x in mid for x in ["moe", "a3b", "-experts-", "experts", "Thinking"]) or m.get('arch', '').endswith('_moe')
        is_a3b = "a3b" in mid

        suitable = False
        reason = ""

        current_cat = categorize_model(m)

        if is_moe:
            if params <= MAX_MOE_TOTAL_PARAMS:
                suitable = True
                reason = "MoE (A3B/Efficient Architecture) - " + ("A3B Priority" if is_a3b else "Compact MoE")
        else:
            if 0 < params <= MAX_DENSE_PARAMS:
                suitable = True
                reason = f"Dense Efficient ({params}B)"
                if params <= ULTRA_LIGHT_THRESHOLD:
                    reason = f"Ultra-Lightweight Specialist ({params}B)"

        if not suitable and params == 0 and m.get('likes', 0) > 100:
            size_str = str(m.get('disk_gb', '0')).replace('GB', '').strip()
            try:
                size = float(size_str) if size_str != 'Unknown' else 0.0
                if 0 < size <= 30:
                    suitable = True
                    reason = "Efficient Footprint (Verified by Size)"
            except: pass

        if suitable:
            if current_cat in shortlist:
                human_ver = str(m['parsed_version'])
                if m['parsed_version'] >= 1000000.0:
                    idx = int(m['parsed_version'] - 1000000.0)
                    if fam in MAJOR_FAM_MAP and idx < len(MAJOR_FAM_MAP[fam]):
                        human_ver = MAJOR_FAM_MAP[fam][idx]

                shortlist[current_cat].append({
                    "model_id": m['id'],
                    "version": human_ver,
                    "params": f"{params}B" if params > 0 else "Unknown",
                    "size_gb": m.get('disk_gb', 'Unknown'),
                    "shortlist_reason": reason,
                    "downloads": m.get('downloads', 0),
                    "likes": m.get('likes', 0)
                })

    for cat in shortlist:
        # Sort by downloads and likes
        shortlist[cat] = sorted(shortlist[cat], key=lambda x: (x['downloads'], x['likes']), reverse=True)

        # SPECIAL BOOST for user requested intelligent micro-models
        if cat == "tool_calling_ultra_light":
            boosted = [m for m in shortlist[cat] if any(x in m['model_id'].lower() for x in ["functiongemma", "granite-4.0"])]
            shortlist[cat] = boosted + [m for m in shortlist[cat] if m not in boosted]

        unique = []
        seen = set()
        for m in shortlist[cat]:
            if m['model_id'] not in seen:
                unique.append(m)
                seen.add(m['model_id'])
        shortlist[cat] = unique[:15]

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(shortlist, f, indent=2)
    print(f"DONE. Efficient Shortlist saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze()
