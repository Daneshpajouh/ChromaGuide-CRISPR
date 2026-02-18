import json
import re

INPUT_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/datasets/GLOBAL_FULL_INVENTORY.json"

# Constraints
MAX_DENSE = 15.0
MAX_MOE_TOTAL = 35.0

# Latest Gen Indicators
LATEST_GEN_REGEX = r"(qwen3|llama-4|gemma-3|granite-4|internvl3\.5|nemotron-3|nemotron-4|r1-distill-qwen3|r1-distill-llama-4)"

SPECIALTIES = {
    "Mathematical / Logical Formalization": ["math", "logic", "proof", "arithmetic", "reasoning"],
    "Vision / Multimodal / OCR": ["vl", "vlm", "vision", "multimodal", "ocr", "internvl"],
    "Technical Coding / Automation": ["code", "coder", "programming"],
    "Medical / Genomic / Science": ["bio", "medical", "gene", "protein", "dna", "science", "physics", "chemistry"],
    "Long-Context Analysis / Retrieval": ["128k", "200k", "512k", "1m", "long-context"],
    "General Purpose Research": ["instruct", "chat", "assistant"]
}

with open(INPUT_FILE, 'r') as f:
    models = json.load(f)

print("# Bleeding-Edge Efficient Specialists (14B Dense / 30B MoE)\n")

for specialty, keywords in SPECIALTIES.items():
    matches = []
    for m in models:
        mid = m['id'].lower()
        tags = [t.lower() for t in m.get('tags', [])]

        # Check if it's the latest generation
        is_latest = re.search(LATEST_GEN_REGEX, mid) or any(re.search(LATEST_GEN_REGEX, t) for t in tags)
        if not is_latest:
            continue

        params = m.get('param_val', 0)
        is_moe = "moe" in mid or "a3b" in mid or m.get('arch', '').endswith('_moe')

        # Apply Efficiency Constraints
        if is_moe:
            if params > MAX_MOE_TOTAL or params == 0: continue
        else:
            if params > MAX_DENSE or params == 0: continue

        # Check for specialty keywords
        if any(k in mid for k in keywords) or any(k in tags for k in keywords):
            matches.append(m)

    # Sort by popularity
    matches = sorted(matches, key=lambda x: (x.get('downloads', 0), x.get('likes', 0)), reverse=True)

    print(f"## {specialty}")
    seen = set()
    count = 0
    for m in matches:
        if count >= 3: break
        if m['id'] in seen: continue
        is_moe_label = " (MoE/A3B)" if ("moe" in m['id'].lower() or "a3b" in m['id'].lower()) else ""
        print(f"- **{m['id']}**: {m.get('params', 'Unknown')} Params{is_moe_label} | {m.get('downloads', 0)} downloads")
        seen.add(m['id'])
        count += 1
    if not matches:
        print("- *No efficient bleeding-edge models found in this category yet.*")
    print("\n")
