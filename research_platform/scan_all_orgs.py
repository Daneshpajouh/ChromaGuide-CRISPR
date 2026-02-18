import requests
import json
import time
import os
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# The User's Definitive List of 76 Orgs
TARGET_ORGS = [
    "databricks", "upstage", "internlm", "01-ai", "Intel", "Salesforce",
    "pollen-robotics", "briaai", "lmstudio-community", "Marvis-AI", "mlx-vision",
    "BAAI", "Ultralytics", "PaddlePaddle", "FunAudioLLM", "meituan-longcat",
    "moondream", "Comfy-Org", "KlingTeam", "arcee-ai", "Wan-AI", "Qwen",
    "open-thoughts", "sentence-transformers", "MiniMaxAI", "cerebras", "moonshotai",
    "allenai", "browser-use", "OpenMed", "Nanbeige", "OmniGen2", "ByteDance-Seed",
    "openbmb", "llamaindex", "FastVideo", "HiDream-ai", "qualcomm", "Zyphra",
    "togethercomputer", "Lightricks", "ggml-org", "LGAI-EXAONE", "inclusionAI",
    "HuggingFaceH4", "LiquidAI", "jinaai", "bigcode", "open-r1", "RedHatAI",
    "baidu", "CohereLabs", "amd", "mistral-community", "perplexity-ai", "amazon",
    "unsloth", "black-forest-labs", "openai", "stabilityai", "HuggingFaceTB",
    "xai-org", "meta-llama", "microsoft", "apple", "zai-org", "tencent", "google",
    "ibm-granite", "nvidia", "XiaomiMiMo", "mistralai", "NexaAI", "deepseek-ai",
    "mlx-community", "huggingface"
]

# Heuristics for <= 30B
# Match explicitly sizes that are safe: 1B, 2B, ... 30B.
# Exclude: 31B+, 70B, 405B, etc.
SIZE_PATTERN = re.compile(r"(\d+(\.\d+)?)[\s\-]?[bB]")

def parse_size(name):
    matches = SIZE_PATTERN.findall(name)
    if not matches:
        return None
    # Take the last match as usually model size is at end or prominent
    try:
        # Check all matches, if any is > 32, reject.
        # But some might be "Layer-12", so purely "B" suffix.
        # The regex enforces B.
        sizes = [float(m[0]) for m in matches]

        # Filter Logic:
        # If ANY size is > 32, it's likely too big (e.g. 70B).
        # Exception: "8x7B" -> 56B (Too big).
        # "8x22B" -> Too big.
        # "4x7B" -> 28B (Safe).

        if any(s > 32 for s in sizes):
            return 999 # Too big

        return max(sizes) # Return the largest safe size found
    except:
        return None

def is_efficient(model):
    name = model.get('modelId', '').split('/')[-1]

    # 1. Explicit Exclusions
    if "70b" in name.lower() or "405b" in name.lower() or "72b" in name.lower():
        return False
    if "gguf" in name.lower() or "awq" in name.lower() or "gptq" in name.lower():
        return True # Quantized versions of large models? Maybe. For now, accept quants.

    # 2. Size Check
    size = parse_size(name)
    if size and size > 32:
        return False

    # 3. Likes Threshold (Quality Filter)
    # If it has < 5 likes, maybe ignore unless it's remarkably new?
    # User checked "Following", so assume all orgs are interesting.
    # But for the manifest, we want "Relevant". Let's keep filter loose for now.
    return True

def scan_org(org):
    print(f"Scanning {org}...")
    url = f"https://huggingface.co/api/models"
    params = {
        "author": org,
        "sort": "likes",
        "direction": "-1",
        "limit": 30
        # "filter": removed to ensure we get results (API might treat comma as AND)
    }

    try:
        # verify=False to bypass local cert issues
        resp = requests.get(url, params=params, timeout=10, verify=False)
        if resp.status_code == 200:
            models = resp.json()
            efficient_models = []
            for m in models:
                if is_efficient(m):
                    efficient_models.append({
                        "name": m['modelId'],
                        "likes": m.get('likes', 0),
                        "downloads": m.get('downloads', 0),
                        "updated": m.get('lastModified', ''),
                        "tags": m.get('tags', [])
                    })
            return efficient_models
        else:
            print(f"Failed {org}: {resp.status_code}")
            return []
    except Exception as e:
        print(f"Error {org}: {e}")
        return []

def main():
    all_data = {}

    for org in TARGET_ORGS:
        models = scan_org(org)
        all_data[org] = models
        time.sleep(0.5) # Polite rate limit

    # Save Master Audit
    with open('/Users/studio/Desktop/PhD/Proposal/research_platform/scans/MASTER_AUDIT_76.json', 'w') as f:
        json.dump(all_data, f, indent=2)

    print("Scan Complete. Saved to runs/MASTER_AUDIT_76.json")

if __name__ == "__main__":
    main()
