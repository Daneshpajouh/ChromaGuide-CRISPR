import json
import re

# Definite hierarchy of generations.
# Higher index = Newer.
GEN_MAP = {
    "llama": ["llama-1", "llama-2", "llama-3", "llama-3.1", "llama-3.2", "llama-3.3", "llama-4"],
    "qwen": ["qwen", "qwen1.5", "qwen2", "qwen2.5", "qwen3"],
    "gemma": ["gemma", "gemma-2", "gemma-3"],
    "phi": ["phi-1", "phi-1.5", "phi-2", "phi-3", "phi-3.5", "phi-4"],
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

    # BigCode
    if "starcoder" in name_lower:
        if "starcoder2" in name_lower: return ("bigcode", 2)
        return ("bigcode", 1)

    # ChatGLM
    if "chatglm" in name_lower or "glm-4" in name_lower:
        if "glm-4" in name_lower: return ("chatglm", 4)
        if "chatglm3" in name_lower: return ("chatglm", 3)
        if "chatglm2" in name_lower: return ("chatglm", 2)
        return ("chatglm", 1)

    # Cohere
    if "aya" in name_lower or "command" in name_lower:
        if "aya-expanse" in name_lower: return ("cohere", 3)
        if "aya-23" in name_lower: return ("cohere", 2)
        if "aya-101" in name_lower: return ("cohere", 1)
        # Command R series
        if "command-r7b" in name_lower: return ("cohere", 3) # Dec 2024
        if "command-r" in name_lower: return ("cohere", 2)
        return ("cohere", 1)

    # Baidu
    if "ernie" in name_lower:
        if "ernie-4.5" in name_lower: return ("baidu", 3)
        if "ernie-4" in name_lower: return ("baidu", 2)
        if "ernie-3" in name_lower: return ("baidu", 1)
        return ("baidu", 0)

    # Qwen Logic
    if "qwen" in name_lower:
        if "qwen3" in name_lower: return ("qwen", 4)
        if "qwen2.5" in name_lower: return ("qwen", 3)
        if "qwen2" in name_lower: return ("qwen", 2)
        if "qwen1.5" in name_lower: return ("qwen", 1)
        return ("qwen", 0)

    # Llama Logic
    if "llama" in name_lower:
        # Llama 4
        if "llama 4" in name_lower or "llama-4" in name_lower: return ("llama", 6)
        # Llama 3
        if "llama-3.3" in name_lower: return ("llama", 5)
        if "llama-3.2" in name_lower: return ("llama", 4)
        if "llama-3.1" in name_lower: return ("llama", 3)
        if "llama-3" in name_lower: return ("llama", 2)
        if "llama-2" in name_lower: return ("llama", 1)
        return ("llama", 0)

    # Gemma Logic
    if "gemma" in name_lower:
        if "gemma-3" in name_lower: return ("gemma", 2)
        if "gemma-2" in name_lower: return ("gemma", 1)
        return ("gemma", 0)

    # Phi Logic
    if "phi" in name_lower:
        if "phi-4" in name_lower: return ("phi", 5)
        if "phi-3.5" in name_lower: return ("phi", 4)
        if "phi-3" in name_lower: return ("phi", 3)
        if "phi-2" in name_lower: return ("phi", 2)
        if "phi-1.5" in name_lower: return ("phi", 1)
        return ("phi", 0)

    # InternLM
    if "internlm" in name_lower:
        if "intern-s1" in name_lower: return ("internlm", 4) # S1 is newest reasoning
        if "internlm3" in name_lower: return ("internlm", 3)
        if "internlm2.5" in name_lower: return ("internlm", 2)
        if "internlm2" in name_lower: return ("internlm", 1)
        return ("internlm", 0)

    return (None, 0)

def main():
    with open('/Users/studio/Desktop/PhD/Proposal/research_platform/scans/SPECIALIZED_AUDIT.json', 'r') as f:
        data = json.load(f)

    # We need to filter the lists inside data[org]

    # 1. Flatten all models to find the "Max Generation" per family
    max_gens = {}
    for org, models in data.items():
        for m in models:
            family, gen = detect_generation(m['name'])
            if family:
                if gen > max_gens.get(family, -1):
                    max_gens[family] = gen

    print("Latest Generations Detected:", max_gens)
    # Example: {'qwen': 3 (2.5), 'llama': 4 (3.2), 'phi': 5 (4)}

    filtered_data = {}

    for org, models in data.items():
        kept_models = []
        for m in models:
            family, gen = detect_generation(m['name'])

            # If it belongs to a known family
            if family:
                # User Policy: "always use the latest versions... instead of using their previous generations"
                # If max_gen is 5 (Phi-4), we should strictly reject Phi-2 (gen 2).
                # But maybe we keep Phi-3 (gen 3) if Phi-4 is not a replacement?
                # The user said "latest available". Usually Phi-4 replaces Phi-3.
                # However, Phi-3-Mini might be unique.
                # Let's be strict. If there is a newer gen, discard older.
                # EXCEPT if the newer gen is NOT "available" or is weird (preview).
                # Let's assume max_gen is the standard.

                # Rule: Keep only if gen >= max_gen - 1?
                # No, user said "only keep the latest versions".
                # So strictly gen == max_gen.
                # UNLESS: max_gen is 'Preview' and we want 'Stable'?
                # Let's enforce gen == max_gen for Llama, Qwen, Gemma.

                if gen < max_gens[family]:
                    # Exception: Size classes.
                    # Qwen2.5-Coder-32B (latest). Qwen2-72B (old).
                    # If this model is older, drop it.
                    continue

            kept_models.append(m)

        if kept_models:
            filtered_data[org] = kept_models

    # Re-generate Markdown
    lines = []
    lines.append("# Specialized Capabilities (Latest Gen Only)")
    lines.append("> **Constraint**: STRICTLY Latest Generations (No Legacy).")
    lines.append(f"> **Families Filtered**: {list(max_gens.keys())}")
    lines.append("")

    # Re-group by Category
    by_category = {}
    KEYWORDS = ["🛡️ Safeguard", "🛠️ Tool Use", "⚡ Edge/Mobile", "🧠 Specialized Instruct"]
    for k in KEYWORDS: by_category[k] = []

    for org, models in filtered_data.items():
        for m in models:
            for cat in m['categories']:
                if cat in by_category:
                    by_category[cat].append(m)

    for cat in KEYWORDS:
        models = by_category[cat]
        if not models: continue

        lines.append(f"## {cat}")
        lines.append("| Model | Likes | Org | Tags |")
        lines.append("|---|---|---|---|")

        models.sort(key=lambda x: x['likes'], reverse=True)
        seen = set()

        for m in models[:25]:
            if m['name'] in seen: continue
            seen.add(m['name'])
            tags_str = ", ".join(m['tags'][:3])
            lines.append(f"| [{m['name'].split('/')[-1]}](https://huggingface.co/{m['name']}) | {m['likes']} | {m['name'].split('/')[0]} | {tags_str} |")
        lines.append("")

    output_path = '/Users/studio/.gemini/antigravity/brain/0d8c8273-fb25-4dde-974e-c7ac7b6d44ec/SPECIALIZED_MANIFEST.md'
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print("Refined Manifest Generated.")

if __name__ == "__main__":
    main()
