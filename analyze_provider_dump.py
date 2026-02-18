import re
import collections

# Define Families to grouping
FAMILIES = {
    "Qwen", "InternVL", "CogVLM", "CogView", "FLUX", "Z-Image", "Molmo",
    "Llama", "DeepSeek", "Moondream", "Gemma", "PaliGemma", "Whisper", "OWSM",
    "Evo", "Hyena", "Caduceus", "Canary", "Lumina"
}

def get_max_version_from_id(model_id):
    # Lowercase
    s = model_id.lower()

    # 1. REMOVE PARAMETER COUNTS to avoid high false positives
    # Remove 7b, 72b, 8b, 0.5b, 100m, 100k
    s = re.sub(r'\d+(\.\d+)?[bmk]', '', s)

    # 2. Extract remaining numbers
    # Look for floats or ints
    candidates = re.findall(r'(\d+(?:\.\d+)?)', s)

    versions = []
    for c in candidates:
        try:
            val = float(c)
            # Filter out likely years/dates if strict
            # But user said "version 2025" might be a thing?
            # Let's keep < 100 for standard versioning, or detect specific version patterns
            if val < 200:
                versions.append(val)
        except: pass

    if not versions:
        return 0.0

    return max(versions)

def run_analysis():
    print("🚀 ANALYZING DUMP FOR MAX VERSIONS...")

    # map family -> list of (version, model_line)
    family_data = collections.defaultdict(list)

    with open("ALL_PROVIDER_MODELS.txt", "r") as f:
        lines = f.readlines()

    for line in lines:
        if "|" not in line: continue
        parts = line.split("|")
        if len(parts) < 2: continue

        model_id = parts[1].strip()

        # Determine Family
        my_family = None
        for fam in FAMILIES:
            if fam.lower() in model_id.lower():
                my_family = fam
                break

        if my_family:
            v_score = get_max_version_from_id(model_id)
            family_data[my_family].append((v_score, line.strip()))

    print("\n🏆 MAX VERSION REPORT (Computed from Raw Dump)")
    print("="*80)

    for fam, entries in family_data.items():
        # Sort by version DESC
        entries.sort(key=lambda x: x[0], reverse=True)

        # Get Top 3 unique versions to show validation
        top_candidates = []
        seen = set()
        for v, l in entries:
            if v not in seen:
                top_candidates.append((v, l))
                seen.add(v)
            if len(top_candidates) >= 3: break

        print(f"\n📂 {fam} (Max Found: v{top_candidates[0][0] if top_candidates else 0})")
        for v, l in top_candidates:
            print(f"   [{v}] {l}")

if __name__ == "__main__":
    run_analysis()
