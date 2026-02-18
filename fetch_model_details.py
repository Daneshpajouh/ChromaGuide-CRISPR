import sys
import re
from huggingface_hub import HfApi

DEFAULT_MODELS = [
    "mlx-community/Qwen3-4B-Instruct-2507-5bit"
]

def get_details(api, model_id):
    try:
        # Fetch siblings explicitly to get file sizes
        info = api.model_info(model_id, files_metadata=True)

        # 1. Modality
        modality = info.pipeline_tag if info.pipeline_tag else "Unknown"

        # 2. Disk Size (GB)
        total_bytes = 0
        if info.siblings:
            for file in info.siblings:
                fname = file.rfilename.lower()
                # Sum common weights
                if fname.endswith(('.safetensors', '.bin', '.pt', '.gguf', '.msgpack', '.model')):
                    if hasattr(file, 'size') and file.size:
                        total_bytes += file.size

        gb_size = total_bytes / (1024**3)
        size_str = f"{gb_size:.2f} GB"

        return size_str, modality

    except Exception as e:
        return "Error", str(e)

def run_details():
    api = HfApi()

    # Use command line args if provided, else defaults
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS

    print(f"{'MODEL ID':<50} | {'STORAGE':<10} | {'MODALITY':<25}")
    print("-" * 90)

    for mid in targets:
        size_gb, mod = get_details(api, mid)
        print(f"{mid:<50} | {size_gb:<10} | {mod:<25}")

if __name__ == "__main__":
    run_details()
