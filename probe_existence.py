from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

CANDIDATES = [
    "black-forest-labs/FLUX.2-dev",
    "THUDM/CogView4-6B",
    "vikhyatk/moondream-v3-preview",
    "mlx-community/InternVL3_5-30B-A3B-4bit",
    "Alpha-VLLM/Lumina-Image-3.0",
    "Tongyi-MAI/Z-Image-Turbo-2",
    "deepseek-ai/DeepSeek-V3.2_bf16",
    "espnet/owsm_v5_medium_1B", # Control (Verified)
    "Lakonik/pi-FLUX.2" # Alt FLUX
]

print("🔍 Probe: Forgotten SOTA List...")

found = []
missing = []

for model_id in CANDIDATES:
    print(f"\nTarget: {model_id}")
    try:
        # Check config or readme
        path = hf_hub_download(repo_id=model_id, filename="config.json")
        print(f"✅ FOUND! {model_id}")
        found.append(model_id)
    except RepositoryNotFoundError:
        # Try readme
        try:
             path = hf_hub_download(repo_id=model_id, filename="README.md")
             print(f"✅ FOUND (Readme Only)! {model_id}")
             found.append(model_id)
        except:
             print(f"❌ MISSING: {model_id}")
             missing.append(model_id)
    except Exception as e:
        print(f"⚠️  Error: {e}")
        if "401" in str(e) or "403" in str(e):
             print(f"🔒 GATED: {model_id}")
             found.append(model_id) # Count as found but locked
        else:
             missing.append(model_id)

print("\n📊 SUMMARY")
print("FOUND:", found)
print("MISSING:", missing)
