import requests
import json
import time
import urllib3
import re

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

KEYWORDS = {
    "🛡️ Safeguard": ["guard", "safety", "moderate", "shield", "defense", "protect", "red-team", "evaluation", "judge"],
    "🛠️ Tool Use": ["function", "tool", "agent", "call", "fc", "api", "action", "planner", "reasoning"],
    "⚡ Edge/Mobile": ["nano", "micro", "tiny", "mobile", "edge", "on-device", "1b", "0.5b", "2b", "small"],
    "🧠 Specialized Instruct": ["math", "code", "medical", "legal", "finance", "science", "biology", "chemistry"]
}

def get_categories(name, tags):
    cats = set()
    text = (name + " " + " ".join(tags)).lower()

    for cat, output_cat in [("guard", "🛡️ Safeguard"), ("function", "🛠️ Tool Use"), ("nano", "⚡ Edge/Mobile")]:
        # Simple check? No, use the detailed dict
        pass

    for label, keywords in KEYWORDS.items():
        for k in keywords:
            if k in text:
                cats.add(label)
                break # Found one keyword for this category, move to next category

    return list(cats)

def scan_org_deep(org):
    print(f"Deep Scanning {org}...")
    url = f"https://huggingface.co/api/models"
    params = {
        "author": org,
        "sort": "likes",
        "direction": "-1",
        "limit": 100, # Expanded Depth
    }

    try:
        resp = requests.get(url, params=params, verify=False, timeout=15)
        if resp.status_code == 200:
            models = resp.json()
            specialized_finds = []

            for m in models:
                name = m['modelId']
                tags = m.get('tags', [])
                categories = get_categories(name, tags)

                # We specifically want models that MATCH one of our special categories
                if categories:
                    # Additional Size Filter: Ignore 70B+ even if safeguard
                    # (User wants lightweight)
                    if "70b" in name.lower() or "405b" in name.lower():
                        continue

                    specialized_finds.append({
                        "name": name,
                        "likes": m.get('likes', 0),
                        "categories": categories,
                        "tags": tags[:5]
                    })
            return specialized_finds
        else:
            return []
    except Exception as e:
        print(f"Error {org}: {e}")
        return []

def main():
    all_specialized = {}

    for org in TARGET_ORGS:
        found = scan_org_deep(org)
        if found:
            all_specialized[org] = found
        time.sleep(0.2)

    # Save Report
    with open('/Users/studio/Desktop/PhD/Proposal/research_platform/scans/SPECIALIZED_AUDIT.json', 'w') as f:
        json.dump(all_specialized, f, indent=2)

    print("Deep Scan Complete. Saved to scans/SPECIALIZED_AUDIT.json")

    # Also generate the Markdown Report immediately
    lines = []
    lines.append("# Specialized Capabilities Manifest")
    lines.append("> **Focus**: Safeguards, Tool Usage, Edge/Mobile, Domain Specific.")
    lines.append("> **Depth**: Top 100 Models per Org managed.")
    lines.append("")

    # Invert the index: Show by Category

    by_category = {k: [] for k in KEYWORDS.keys()}

    for org, models in all_specialized.items():
        for m in models:
            for cat in m['categories']:
                by_category[cat].append(m)

    for cat, models in by_category.items():
        if not models:
            continue

        lines.append(f"## {cat}")
        lines.append("| Model | Likes | Org | Tags |")
        lines.append("|---|---|---|---|")

        # Sort by likes
        models.sort(key=lambda x: x['likes'], reverse=True)

        # Deduplicate by name
        seen = set()

        for m in models[:20]: # Top 20 per category
            if m['name'] in seen: continue
            seen.add(m['name'])

            org_name = m['name'].split('/')[0]
            clean_name = m['name'].split('/')[-1]
            tags_str = ", ".join(m['tags'][:3])
            lines.append(f"| [{clean_name}](https://huggingface.co/{m['name']}) | {m['likes']} | {org_name} | {tags_str} |")
        lines.append("")

    with open('/Users/studio/.gemini/antigravity/brain/0d8c8273-fb25-4dde-974e-c7ac7b6d44ec/SPECIALIZED_MANIFEST.md', 'w') as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
