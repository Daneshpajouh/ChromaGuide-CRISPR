import re
import csv
import time
import math
from datetime import datetime
from typing import List, Dict, Set
from huggingface_hub import HfApi

class UniversalAuditor:
    """
    The Ultimate SOTA Auditor.
    finds ALL organizations, ALL categories, Top 100 per category.
    """

    def __init__(self):
        self.api = HfApi()
        self.orgs_found: Set[str] = set()

    TASKS = [
        # Vision
        "image-text-to-text", "text-to-image", "image-to-text", "video-text-to-text",
        "text-to-video", "image-to-video", "video-classification", "object-detection",
        "image-segmentation", "depth-estimation", "zero-shot-image-classification",
        "mask-generation",
        # Audio
        "automatic-speech-recognition", "text-to-speech", "audio-text-to-text",
        "audio-classification", "voice-activity-detection",
        # Text/Code
        "text-generation", "text-classification", "token-classification",
        "question-answering", "summarization", "table-question-answering",
        "document-question-answering",
        # Science/Other
        "reinforcement-learning", "robotics", "graph-ml", "tabular-classification"
    ]

    def extract_metadata(self, model) -> Dict:
        """Parse strict metadata from model object."""
        mid = model.id

        # 1. Version
        ver = 0.0
        # Clean naming
        clean_id = re.sub(r'\d+(b|m|k)', '', mid.lower()) # remove params
        matches = re.findall(r'v(\d+(?:\.\d+)?)', clean_id)
        if not matches: matches = re.findall(r'(\d+(?:\.\d+)?)', clean_id)
        valid_vers = [float(x) for x in matches if float(x) < 50] # Filter dates
        if valid_vers: ver = max(valid_vers)

        # 2. Parameters
        params = "Unknown"
        p_match = re.search(r'(\d+(?:\.\d+)?)(b|m)', mid.lower())
        if p_match:
            params = f"{p_match.group(1).upper()}{p_match.group(2).upper()}"

        # 3. Size (Estimate from safetensors if available in siblings - fast check)
        # Note: full=True in list_models gives siblings!
        size_gb = 0.0
        if hasattr(model, 'siblings') and model.siblings:
            for s in model.siblings:
                if s.rfilename.endswith(('.safetensors', '.bin', '.pt', '.gguf')):
                    if hasattr(s, 'size') and s.size: size_gb += s.size
        size_gb = size_gb / (1024**3)

        return {
            "id": mid,
            "author": mid.split('/')[0] if '/' in mid else "Unknown",
            "version": ver,
            "params": params,
            "size_gb": round(size_gb, 2),
            "date": str(model.created_at)[:10] if model.created_at else "N/A",
            "downloads": model.downloads,
            "likes": model.likes,
            "task": model.pipeline_tag
        }

    def discover_orgs(self):
        print("🔍 Dynamically Discovering Active Organizations...")
        # Scan recent models to find active orgs
        models = self.api.list_models(sort="createdAt", direction="-1", limit=2000)
        for m in models:
            if '/' in m.id:
                self.orgs_found.add(m.id.split('/')[0])
        print(f"   -> Found {len(self.orgs_found)} Active Organizations.")

        # Export Org List
        with open("organization_atlas.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(["Organization"])
            for o in sorted(list(self.orgs_found)): w.writerow([o])

    def run_audit(self):
        print(f"🚀 STARTING UNIVERSAL AUDIT across {len(self.TASKS)} Categories...")

        all_results = []

        for task in self.TASKS:
            print(f"\n📂 CATEGORY: {task}")
            try:
                # Fetch Top 500 by Date (Newest first)
                models = self.api.list_models(filter=task, sort="createdAt", direction="-1", limit=500, full=True)

                # Process
                processed = []
                for m in models:
                    meta = self.extract_metadata(m)

                    # Filtering Logic for "Top 100"
                    # We want: Max Version > High Activity
                    # Heuristic Score
                    version_score = meta['version'] * 100
                    like_score = 0
                    if meta['likes']: like_score = math.log(meta['likes'] + 1)

                    # Combined Rank
                    meta['score'] = version_score + like_score
                    processed.append(meta)

                # Sort by Score DESC
                processed.sort(key=lambda x: x['score'], reverse=True)

                # Keep Top 100
                top_100 = processed[:100]
                all_results.extend(top_100)

                print(f"   -> Retained {len(top_100)} candidates.")
                if top_100:
                    print(f"      Top: {top_100[0]['id']} (v{top_100[0]['version']}, {top_100[0]['size_gb']}GB)")

            except Exception as e:
                print(f"   -> Error scanning {task}: {e}")

        # Export Comprehensive CSV
        print("\n💾 Exporting UNIVERSAL_SOTA_RESULTS.csv...")
        keys = ["task", "id", "author", "version", "params", "size_gb", "date", "downloads", "likes", "score"]
        with open("universal_sota_results.csv", "w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)
        print("✅ Done.")

if __name__ == "__main__":
    auditor = UniversalAuditor()
    auditor.discover_orgs()
    auditor.run_audit()
