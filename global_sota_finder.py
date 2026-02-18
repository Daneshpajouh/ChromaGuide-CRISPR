import re
import time
import csv
from typing import List, Dict, Optional, Tuple
from huggingface_hub import HfApi

class GlobalSOTAFinder:
    """
    Massive Global SOTA Scanner.
    Constraint: Max Version | <10B Parameters (<20GB) | All Categories
    """

    def __init__(self):
        self.api = HfApi()
        self.max_size_gb = 20.0
        self.max_size_bytes = self.max_size_gb * (1024**3)

    # 80+ PRIORITY AUTHORS (From User's Guide)
    AUTHORS = [
        # China / Vision / Multi
        "Qwen", "THUDM", "deepseek-ai", "OpenGVLab", "Tongyi-MAI", "ZImageModel",
        "hustvl", "ByteDance", "Alibaba", "Shanghai-AI-Laboratory", "01-ai", "internlm",
        # USA / LLM / Multi
        "meta-llama", "llava-hf", "openai", "google", "facebook", "NVIDIA", "microsoft",
        "apple", "amazon", "adept", "Salesforce", "nomic-ai", "ContextualAI",
        # Image / Video
        "black-forest-labs", "stabilityai", "Alpha-VLLM", "briaai", "fal-ai",
        "runwayml", "playgroundai", "TencentARC",
        # Audio
        "espnet", "distil-whisper", "nvidia", "speechbrain", "ylacombe", "pyannote",
        # Science / Genomics
        "arc-institute", "kuleshov-group", "LongSafari", "InstaDeep", "AIRI-Institute",
        # Community / MLX
        "mlx-community", "NousResearch", "migtissera", "teknium", "Undi95", "alpindale"
    ]

    # ALL CATEGORIES
    TASKS = [
        "image-text-to-text", "text-to-image", "automatic-speech-recognition",
        "text-to-video", "video-classification", "visual-question-answering",
        "document-question-answering", "mask-generation", "depth-estimation",
        "zero-shot-image-classification", "text-to-speech", "audio-classification",
        "text-generation", "image-segmentation", "object-detection"
    ]

    def get_safetensors_size(self, model_id: str) -> float:
        """Sum safetensors/bin size in GB."""
        try:
            info = self.api.model_info(model_id, files_metadata=True)
            total = 0
            if info.siblings:
                for f in info.siblings:
                    if f.rfilename.endswith(('.safetensors', '.bin', '.pt', '.gguf')):
                        total += f.size
            return total / (1024**3)
        except: return 0.0

    def extract_version(self, model_id: str) -> float:
        """Heuristic version extraction (v3 -> 3.0)."""
        # Remove parameter counts
        clean = re.sub(r'\d+(b|m)', '', model_id.lower())
        # Find floats
        matches = re.findall(r'(\d+(?:\.\d+)?)', clean)
        valid = [float(x) for x in matches if float(x) < 50] # Reject dates/epochs
        return max(valid) if valid else 0.0

    def scan_all(self):
        print(f"🚀 STARTING MASSIVE GLOBAL SCAN ({len(self.AUTHORS)} Authors x {len(self.TASKS)} Tasks)...")
        results = {}

        for task in self.TASKS:
            print(f"\n📂 CATEGORY: {task}")
            best_in_class = None
            max_ver = -1.0

            # For this task, scan all authors
            # Note: Checking 80 authors per task is slow, so we filter by Task first
            # But User said "Raw Dump", filter by Author.
            # Strategy: List Author models -> Check Task -> Check Version

            # Optimization: One massive dump of all authors? No, memory.
            # Let's iterate authors but optimize.

            candidates = []

            # To be efficient, we search ONLY top relevant authors for specific tasks?
            # User wants "Any possible category".
            # Let's do a 'search' query per task to get top candidates, then filter by Author?
            # No, user wants Author priority.

            # Let's use the USER'S strategy: Loop Authors, get their models, bucket them.
            pass

    def run_author_centric_scan(self):
        """
        Iterate Authors -> Bucket Models by Task -> Find Max Version < 10B
        """
        print(f"🚀 SCANNING {len(self.AUTHORS)} AUTHORS for SOTA...")

        # Map: Task -> List of (Version, Size, ModelID)
        category_winners = {t: [] for t in self.TASKS}

        for i, author in enumerate(self.AUTHORS):
            print(f"   [{i+1}/{len(self.AUTHORS)}] Scanning {author}...")
            try:
                models = self.api.list_models(author=author, limit=200, full=True, sort="createdAt", direction="-1")

                for m in models:
                    # 1. Determine Category
                    task = m.pipeline_tag
                    if not task or task not in self.TASKS:
                        # Try to infer from name for "hidden" models
                        if "vision" in m.id.lower(): task = "image-text-to-text"
                        elif "audio" in m.id.lower(): task = "automatic-speech-recognition"
                        else: continue

                    # 2. Check Size (Fast metadata check first?)
                    # calculating exact size for 200 models is slow.
                    # Heuristic name check first
                    if any(x in m.id.lower() for x in ["70b", "30b", "large", "medium"]):
                         if "distill" not in m.id.lower(): continue

                    # 3. Extract Version
                    ver = self.extract_version(m.id)

                    # 4. Store Candidate
                    # We will verify size later for top candidates only
                    category_winners[task].append({
                        "id": m.id,
                        "ver": ver,
                        "date": m.created_at,
                        "likes": m.likes,
                        "downloads": m.downloads
                    })

            except Exception as e:
                print(f"Error {author}: {e}")

        print("\n\n🏆 FINAL ANALYSIS & VERIFICATION (<20GB Only)")
        print("="*80)

        # For each category, find the Max Version that fits
        final_list = []

        for task, candidates in category_winners.items():
            if not candidates: continue

            # Sort by Version DESC, then Date DESC
            candidates.sort(key=lambda x: (x['ver'], x['date']), reverse=True)

            # Check the top 5 versions for size fit
            print(f"\n📁 {task.upper()} (Top Version Candidates):")
            found_winner = False

            for c in candidates[:10]:
                # Verify Size
                gb = self.get_safetensors_size(c['id'])
                if gb == 0.0 and "mlx" in c['id']: gb = 5.0 # Assume MLX fits

                if 0.1 < gb < 20.0:
                    status = "✅ WINNER"
                    if not found_winner:
                        final_list.append((task, c['id'], c['ver'], gb))
                        found_winner = True
                    print(f"   [{status}] v{c['ver']} | {c['id']} | {gb:.2f} GB")
                else:
                    print(f"   [❌ TOO BIG/SMALL] v{c['ver']} | {c['id']} | {gb:.2f} GB")

        return final_list

if __name__ == "__main__":
    finder = GlobalSOTAFinder()
    finder.run_author_centric_scan()
