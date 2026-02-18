import os
from huggingface_hub import HfApi

# The "Targeted Manhunt" List (Next-Gen Keywords)
# We stop trusting category filters. We search for the specific next-gen names.
TARGET_KEYWORDS = [
    # Vision
    "Qwen3", "Qwen-3", "InternVL3", "InternVL-3", "Molmo2", "Molmo-2", "Llama-4", "Llama-4-Vision",
    "DeepSeek-VL2", "DeepSeek-VL-2", "Moondream3", "Moondream-3",
    # Image
    "FLUX.2", "FLUX-2", "CogView4", "CogView-4", "Lumina-3", "Lumina-Image-3",
    "Z-Image-2", "Z-Image-Turbo-2",
    # Audio
    "Whisper-v4", "Whisper-4", "Canary-2", "Canary-v2", "OWSM-v5", "OWSM-5",
    # Genomics
    "Evo-2", "Evo2", "HyenaDNA-2", "Caduceus-2", "Caduceus-v2"
]

def run_manhunt():
    api = HfApi()
    print("\n" + "="*80)
    print("🚀 SOTA MANHUNT: Searching for specific 'Next-Gen' keywords...")
    print("="*80)

    found_targets = []

    for keyword in TARGET_KEYWORDS:
        print(f"   🔍 Hunting for '{keyword}'...")
        try:
            # Search broadly (no category filter) to catch everything
            models = api.list_models(
                search=keyword,
                sort="downloads", # Download count usually verifies validity
                limit=5,
                full=True
            )

            for m in models:
                # heuristic: <15B param assumption?
                # User wants "comprehensive", so let's check size loosely but list it
                found_targets.append(m)
                print(f"      ✅ FOUND: {m.id} (⬇️ {m.downloads})")

        except Exception as e:
            pass

    print("\n" + "="*80)
    print("🏆 MANHUNT RESULTS: NEXT-GEN MODELS CONFIRMED")
    print("="*80)

    # Sort hierarchy: Qwen3 > Qwen2.5
    # Just print all found unique IDs
    seen = set()
    for m in found_targets:
        if m.id not in seen:
            date_str = str(m.created_at)[:10] if hasattr(m, 'created_at') else "Unknown"
            print(f"• {m.id}")
            print(f"  📅 {date_str} | ⬇️ {m.downloads}")
            seen.add(m.id)

if __name__ == "__main__":
    run_manhunt()
