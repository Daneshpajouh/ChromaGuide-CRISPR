import json
import re
import os

def parse_size_strict(name):
    # Regex to find parameter counts like 7b, 8B, 1.5b, etc.
    # We want to exclude anything > 32B
    matches = re.findall(r"(\d+(\.\d+)?)[\s\-]?[bB]", name, re.IGNORECASE)
    if not matches:
        return None
    try:
        # Get largest number found
        sizes = [float(m[0]) for m in matches]
        max_size = max(sizes)
        return max_size
    except:
        return None

def is_truly_efficient(model):
    name = model['name']
    size = parse_size_strict(name)

    # 1. Size Check
    if size:
        if size > 32:
            return False
    else:
        # No size in name?
        # Check for blacklist keywords
        bad_keywords = ["70b", "405b", "132b", "command-r-plus", "large", "medium"]
        # "large" is subjective, but "mistral-large" is big. "bert-large" is small.
        # Let's stick to explicit huge sizes.
        if "70b" in name.lower() or "405b" in name.lower() or "command-r-plus" in name.lower():
            return False
        # DBRX check
        if "dbrx" in name.lower() and "instruct" in name.lower():
            # DBRX is 132B
            return False

    # 2. Keywords for Efficiency
    # We want to keep it if it's explicitly efficient OR just standard sized (7B)
    return True

def main():
    json_path = '/Users/studio/Desktop/PhD/Proposal/research_platform/scans/MASTER_AUDIT_76.json'
    with open(json_path, 'r') as f:
        data = json.load(f)

    lines = []
    lines.append("# Global Model Manifest: The Efficient Frontier")
    lines.append("> **Scope**: 76 Organizations Scanned via API.")
    lines.append("> **Constraint**: <= 32B Parameters (approx).")
    lines.append("> **Date**: December 2025")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("This manifest covers the 'Long Tail' of AI innovation, highlighting high-efficiency models from Asia, Europe, and the US that rival the Titans.")
    lines.append("")

    lines.append("## Top Findings by Organization")
    lines.append("| Organization | Top Model | Est. Params | Likes | Description/Tags |")
    lines.append("|---|---|---|---|---|")

    # Process each org
    # Sort orgs by total likes of their top model to surface important ones first? Or alphabetical?
    # Let's sort by "Importance" (max likes of top model)

    org_stats = []
    for org, models in data.items():
        if not models:
            continue

        # Filter models
        valid_models = [m for m in models if is_truly_efficient(m)]
        if not valid_models:
            continue

        # Sort by likes
        valid_models.sort(key=lambda x: x['likes'], reverse=True)
        top_model = valid_models[0]

        org_stats.append({
            "org": org,
            "top_model": top_model,
            "count": len(valid_models)
        })

    # Sort orgs by top model likes
    org_stats.sort(key=lambda x: x['top_model']['likes'], reverse=True)

    for item in org_stats:
        org = item['org']
        m = item['top_model']
        name = m['name']
        likes = m['likes']

        # Estimate params for display
        size = parse_size_strict(name)
        size_str = f"{size}B" if size else "Unknown"

        # Tags formatting
        tags = m.get('tags', [])
        # Extract interesting tags
        interesting = [t for t in tags if t not in ["transformers", "safetensors", "text-generation", "region:us", "license:apache-2.0", "custom_code", "pytorch", "en"]]
        tags_str = ", ".join(interesting[:3])

        lines.append(f"| **{org}** | [{name}](https://huggingface.co/{name}) | {size_str} | {likes} | {tags_str} |")

    # Write Manifest
    output_path = '/Users/studio/.gemini/antigravity/brain/0d8c8273-fb25-4dde-974e-c7ac7b6d44ec/GLOBAL_MODEL_MANIFEST.md'
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Manifest written to {output_path}")

if __name__ == "__main__":
    main()
