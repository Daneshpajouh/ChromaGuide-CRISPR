import re
import csv
import time
import math
from typing import List, Dict, Set
from huggingface_hub import HfApi

class UltimateAuditorV2:
    """
    The Omniscient SOTA Auditor V2.
    Dimensions: Tasks (30+), Libraries (20+), Apps (30+), Providers (20+), Misc (10+).
    Output: Top 100 Per Dimension with Rich Metadata.
    """

    def __init__(self):
        self.api = HfApi()
        self.orgs_found: Set[str] = set()

    # --- DIMENSION 1: TASKS ---
    TASKS = [
        "image-text-to-text", "text-to-image", "image-to-text", "video-text-to-text",
        "text-to-video", "image-to-video", "video-classification", "object-detection",
        "image-segmentation", "depth-estimation", "zero-shot-image-classification",
        "mask-generation", "automatic-speech-recognition", "text-to-speech",
        "audio-text-to-text", "voice-activity-detection", "text-generation",
        "text-classification", "token-classification", "question-answering",
        "summarization", "table-question-answering", "document-question-answering",
        "reinforcement-learning", "robotics", "graph-ml", "tabular-classification",
        "time-series-forecasting", "visual-question-answering", "unconditional-image-generation"
    ]

    # --- DIMENSION 2: LIBRARIES ---
    LIBRARIES = [
        "pytorch", "tensorflow", "jax", "transformers", "diffusers", "safetensors",
        "onnx", "gguf", "mlx", "timm", "sentence-transformers", "open_clip",
        "adapters", "peft", "flax", "scikit-learn", "spacy", "asteroid", "espnet",
        "speechbrain", "nemo"
    ]

    # --- DIMENSION 3: APPS (Mapped to Tags/Search) ---
    APPS = {
        "llama.cpp": "llama-cpp",
        "LM Studio": "lm-studio", # Search if tag fails
        "Jan": "jan",
        "Backyard AI": "backyard",
        "Draw Things": "draw-things",
        "DiffusionBee": "diffusionbee",
        "Jellybox": "jellybox",
        "RecurseChat": "recursechat",
        "Msty": "msty",
        "Sanctum": "sanctum",
        "JoyFusion": "joyfusion",
        "LocalAI": "localai",
        "vLLM": "vllm",
        "node-llama-cpp": "node-llama-cpp",
        "Ollama": "ollama",
        "TGI": "text-generation-inference",
        "MLX LM": "mlx-lm", # Specific MLX tag
        "Docker Model Runner": "docker", # Keyword
        "Lemonade": "lemonade"
    }

    # --- DIMENSION 4: PROVIDERS (Mapped to Search/Tags) ---
    PROVIDERS = {
        "Groq": "groq",
        "Novita": "novita",
        "Nebius AI": "nebius",
        "Cerebras": "cerebras",
        "SambaNova": "sambanova",
        "Nscale": "nscale",
        "fal": "fal",
        "Hyperbolic": "hyperbolic",
        "Together AI": "together",
        "Fireworks": "fireworks",
        "Featherless AI": "featherless",
        "Zai": "zai",
        "Replicate": "replicate",
        "Cohere": "cohere",
        "Scaleway": "scaleway",
        "Public AI": "public-ai",
        "OVHcloud": "ovh",
        "HF Inference API": "inference-api",
        "WaveSpeed": "wavespeed"
    }

    # --- DIMENSION 5: MISC / TAGS ---
    MISC_TAGS = {
        "Inference Endpoints": "inference-endpoints",
        "Eval Results": "eval-results", # often in metadata
        "Merge": "merge",
        "4-bit": "4-bit",
        "8-bit": "8-bit",
        "custom_code": "custom_code",
        "text-embeddings-inference": "text-embeddings-inference",
        "Mixture of Experts": "moe", # or 'mixture-of-experts'
        "Carbon Emissions": "carbon" # meta search
    }

    def extract_metadata(self, model, category_type="Task", category_name="Unknown") -> Dict:
        """Parse strict metadata."""
        mid = model.id

        # Version
        ver = 0.0
        clean_id = re.sub(r'\d+(b|m|k)', '', mid.lower())
        matches = re.findall(r'v(\d+(?:\.\d+)?)', clean_id)
        if not matches: matches = re.findall(r'(\d+(?:\.\d+)?)', clean_id)
        valid_vers = [float(x) for x in matches if float(x) < 50]
        if valid_vers: ver = max(valid_vers)

        # Params
        params = "Unknown"
        p_match = re.search(r'(\d+(?:\.\d+)?)(b|m)', mid.lower())
        if p_match: params = f"{p_match.group(1).upper()}{p_match.group(2).upper()}"

        # Size (GB)
        size_gb = 0.0
        if hasattr(model, 'siblings') and model.siblings:
            for s in model.siblings:
                if s.rfilename.endswith(('.safetensors', '.bin', '.pt', '.gguf')):
                    if hasattr(s, 'size') and s.size: size_gb += s.size
        size_gb = round(size_gb / (1024**3), 2)

        # Tags formatting
        tags = model.tags if model.tags else []

        return {
            "DIMENSION_TYPE": category_type,
            "DIMENSION_NAME": category_name,
            "id": mid,
            "author": mid.split('/')[0] if '/' in mid else "Unknown",
            "version": ver,
            "params": params,
            "size_gb": size_gb,
            "date": str(model.created_at)[:10] if model.created_at else "N/A",
            "downloads": model.downloads,
            "likes": model.likes,
            "tags": str(tags)[:100]
        }

    def run_ultimate_audit(self):
        print("🚀 STARTING THE ULTIMATE MULTI-DIMENSIONAL AUDIT V2...")
        all_results = []

        # 1. SCAN TASKS
        for task in self.TASKS:
            print(f"📂 Scanning Task: {task}...")
            try:
                models = self.api.list_models(filter=task, sort="createdAt", direction="-1", limit=50, full=True)
                for m in models:
                    all_results.append(self.extract_metadata(m, "Task", task))
                    if '/' in m.id: self.orgs_found.add(m.id.split('/')[0])
            except: pass

        # 2. SCAN LIBRARIES
        for lib in self.LIBRARIES:
            print(f"📚 Scanning Library: {lib}...")
            try:
                models = self.api.list_models(filter=lib, sort="createdAt", direction="-1", limit=50, full=True)
                for m in models:
                    all_results.append(self.extract_metadata(m, "Library", lib))
                    if '/' in m.id: self.orgs_found.add(m.id.split('/')[0])
            except: pass

        # 3. SCAN APPS
        for app_name, tag in self.APPS.items():
            print(f"📱 Scanning App: {app_name} ({tag})...")
            try:
                # Try tag
                models = self.api.list_models(filter=tag, sort="createdAt", direction="-1", limit=50, full=True)
                # Fallback Search
                if not models:
                    models = self.api.list_models(search=app_name, sort="createdAt", direction="-1", limit=50, full=True)

                for m in models:
                    all_results.append(self.extract_metadata(m, "App", app_name))
                    if '/' in m.id: self.orgs_found.add(m.id.split('/')[0])
            except: pass

        # 4. SCAN PROVIDERS
        for prov_name, tag in self.PROVIDERS.items():
            print(f"☁️ Scanning Provider: {prov_name} ({tag})...")
            try:
                models = self.api.list_models(search=prov_name, sort="createdAt", direction="-1", limit=50, full=True)
                for m in models:
                    all_results.append(self.extract_metadata(m, "Provider", prov_name))
                    if '/' in m.id: self.orgs_found.add(m.id.split('/')[0])
            except: pass

        # 5. SCAN MISC TAGS
        for misc_name, tag in self.MISC_TAGS.items():
            print(f"🏷️ Scanning Misc: {misc_name} ({tag})...")
            try:
                models = self.api.list_models(filter=tag, sort="createdAt", direction="-1", limit=50, full=True)
                for m in models:
                    all_results.append(self.extract_metadata(m, "Misc", misc_name))
                    if '/' in m.id: self.orgs_found.add(m.id.split('/')[0])
            except: pass

        # EXPORT CSV
        print("\n💾 Exporting ULTIMATE_MATRIX_V2.csv...")
        keys = ["DIMENSION_TYPE", "DIMENSION_NAME", "id", "author", "version", "params", "size_gb", "date", "downloads", "likes", "tags"]
        with open("ultimate_matrix_v2.csv", "w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)

        # EXPORT ORGS
        print("💾 Exporting ORGANIZATION_UNIVERSE_V2.csv...")
        with open("organization_universe_v2.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(["Organization"])
            for o in sorted(list(self.orgs_found)): w.writerow([o])

        print("✅ Ultimate Audit V2 Complete.")

if __name__ == "__main__":
    auditor = UltimateAuditorV2()
    auditor.run_ultimate_audit()
