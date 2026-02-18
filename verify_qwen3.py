from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

CANDIDATES = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct-GGUF",
    "NexaAI/Qwen3-VL-8B-Instruct-GGUF",
    "Qwen/Qwen2.5-VL-7B-Instruct" # Control
]

print("🔍 Verifying Qwen3 Existence...")

for model_id in CANDIDATES:
    print(f"\nTarget: {model_id}")
    try:
        path = hf_hub_download(repo_id=model_id, filename="config.json")
        print(f"✅ FOUND! Config downloaded to {path}")
    except RepositoryNotFoundError:
        print("❌ Not Found (404)")
    except Exception as e:
        print(f"⚠️  Error: {e}")
        if "401" in str(e):
             print("🔒 Gated/Private (Exists but locked)")
