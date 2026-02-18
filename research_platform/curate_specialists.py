import json

INPUT_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/datasets/GLOBAL_FULL_INVENTORY.json"

TASKS = {
    "Mathematical Formalization": ["math", "logic", "proof"],
    "Scientific Vision / Multimodal": ["vision", "vl", "vlm", "multimodal"],
    "Advanced Coding": ["code", "coder"],
    "Long-Context Synthesis": ["128k", "200k", "1m", "long-context"],
    "Biomedical / Genomic": ["bio", "medical", "clinical", "biology", "dna", "protein"],
    "Reasoning & Thinking": ["thought", "thinking", "reasoning", "r1", "qwq"]
}

with open(INPUT_FILE, 'r') as f:
    models = json.load(f)

print("# Specialist Models for PhD Tasks\n")

for task, keywords in TASKS.items():
    specialists = []
    for m in models:
        mid = m['id'].lower()
        tags = [t.lower() for t in m.get('tags', [])]
        if any(k in mid for k in keywords) or any(k in tags for k in keywords):
            # Exclude those that are clearly not main models or are too old if we can
            specialists.append(m)

    # Sort by downloads and likes
    specialists = sorted(specialists, key=lambda x: (x.get('downloads', 0), x.get('likes', 0)), reverse=True)

    print(f"## {task}")
    seen = set()
    count = 0
    for m in specialists:
        if count >= 3: break
        if m['id'] in seen: continue
        print(f"- **{m['id']}**: {m.get('params', 'Unknown params')} | {m.get('downloads', 0)} downloads")
        seen.add(m['id'])
        count += 1
    print("\n")
