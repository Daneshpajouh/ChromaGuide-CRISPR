import os
from huggingface_hub import HfApi

# Confirmed Official Org IDs from User's Guide + Massive Expansion
PROVIDERS = [
    # --- CORE COMMUNITY (User Requested) ---
    "mlx-community",

    # --- VISION / MULTIMODAL LEADING LABS ---
    "Qwen", "OpenGVLab", "THUDM", "deepseek-ai", "jinaai", "allenai",
    "hustvl", "llava-hf", "meta-llama", "google", "microsoft", "vikhyatk",
    "adept", "HuggingFaceM4", "idefics-80b-team", "visheratin",
    "Salesforce", "nomic-ai", "BAAI", # BAAI for Emu/Bunny

    # --- IMAGE / VIDEO GENERATION ---
    "black-forest-labs", "Tongyi-MAI", "ZImageModel", "ByteDance", "Alpha-VLLM",
    "stabilityai", "briaai", "fal-ai", "segmind", "kandinsky-community",
    "TencentARC", "playgroundai", "runwayml", "lucataco",

    # --- GENOMICS / SCIENCE ---
    "arc-institute", "kuleshov-group", "LongSafari", "InstaDeep", "AIRI-Institute",
    "zhihan1996", "beyoru", "songlab", "GatorTron", "recursionpharma",

    # --- AUDIO / SPEECH ---
    "openai", "facebook", "nvidia", "espnet", "distil-whisper", "speechbrain",
    "ylacombe", "collabora", "pyannote",

    # --- GLOBAL LLM POWERHOUSES (Foundations likely to have Multi-modal variants) ---
    "mistralai", "01-ai", "tiiuae", # Falcon
    "upstage", # Solar
    "CohereForAI", "databricks", "mosaicml", "togethercomputer",
    "apple", "amazon", "ibm-granite", "ContextualAI", "Deci", "abacusai",
    "Writer", "Nexusflow", "defog", "Phind",

    # --- TOP TIER FINE-TUNERS & AGGREGATORS ---
    "NousResearch", "OpenChat", "teknium", "migtissera", "alpindale",
    "Undi95", "Gryphe", "Sao10K", "Xwin-LM", "cognitivecomputations"
]

def run_dump():
    api = HfApi()
    output_file = "ALL_PROVIDER_MODELS.txt"

    print(f"🚀 STARTING RAW DUMP of {len(PROVIDERS)} Providers...")

    with open(output_file, "w") as f:
        f.write(f"ORG | MODEL_ID | CREATED_AT | DOWNLOADS\n")
        f.write("="*80 + "\n")

        for org in PROVIDERS:
            print(f"   📥 Fetching full catalog for: {org}...")
            try:
                # Fetch ALL models for this author (no limits, no filters)
                models = api.list_models(author=org, full=True, sort="createdAt", direction="-1")

                count = 0
                for m in models:
                    date_str = str(m.created_at)[:10] if hasattr(m, 'created_at') else "Unknown"
                    # Write pure raw line
                    f.write(f"{org} | {m.id} | {date_str} | {m.downloads}\n")
                    count += 1

                print(f"      -> Retrieved {count} models.")

            except Exception as e:
                print(f"      -> Error fetching {org}: {e}")
                f.write(f"{org} | ERROR_FETCHING | - | -\n")

    print(f"\n✅ DUMP COMPLETE. Saved to {output_file}")

if __name__ == "__main__":
    run_dump()
