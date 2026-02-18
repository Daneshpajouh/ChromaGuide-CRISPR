from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

CANDIDATES = [
    "Lakonik/pi-FLUX.2", # Potential v2 candidate
    "city96/FLUX.1-schnell-gguf", # Current
    "black-forest-labs/FLUX.1-schnell", # Official
    "Shakker-Labs/FLUX.1-dev-LoRA-add-details" # Often called v2 by users
]

print("🔍 Verifying FLUX Candidates...")

for model_id in CANDIDATES:
    print(f"\nTarget: {model_id}")
    try:
        # Check for config or readme
        path = hf_hub_download(repo_id=model_id, filename="README.md")
        print(f"✅ FOUND! Readme downloaded to {path}")

        # Check for GGUF if possible
        if "gguf" in model_id.lower():
            try:
                # Try to list files to find a GGUF
                from huggingface_hub import list_repo_files
                files = list_repo_files(model_id)
                ggufs = [f for f in files if f.endswith(".gguf")]
                if ggufs:
                    print(f"   --> Found GGUFs: {ggufs[:3]}")
                else:
                    print("   --> No GGUF files found.")
            except:
                pass

    except RepositoryNotFoundError:
        print("❌ Not Found (404)")
    except Exception as e:
        print(f"⚠️  Error: {e}")
