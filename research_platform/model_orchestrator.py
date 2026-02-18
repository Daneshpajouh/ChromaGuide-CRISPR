import os
import subprocess
import json
import asyncio
import time
from typing import List, Optional

class ModelOrchestrator:
    """
    The brain of the PhD Research Platform.
    Dynamically orchestrates Local SOTA models across Deep, Regular, and Fast tiers.
    Optimized for Mac Studio M3 Ultra with 'Elements' SSD storage.
    """

    def __init__(self):
        self.manifest_path = "research_platform/MODEL_MANIFEST.json"
        self.output_dir = "research_platform/output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_manifest()

    def load_manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

    def get_model_path(self, category: str, tier: str = "deep_apex") -> Optional[str]:
        """
        Locate the local path for a specific model category and tier.
        Supports: deep_apex, regular_scientific, ultra_light_fast.
        """
        self.load_manifest()
        base_dir = self.manifest.get("base_model_dir", "/Volumes/Elements/research_hub_models")

        # Mapping tier names to manifest keys if they differ
        models = self.manifest.get(tier, [])
        for m in models:
            if m['category'] == category:
                return os.path.join(base_dir, m['local_path'])
        return None

    async def query_async(self, prompt: str, tier: str = "deep_apex", category: str = "unified", model_type: Optional[str] = None, max_tokens: int = 1000) -> str:
        """
        Execute a local MLX or GGUF query asynchronously.
        """
        final_category = model_type if model_type else category
        model_path = self.get_model_path(final_category, tier)
        if not model_path:
            return f"Error: Model category '{final_category}' not found in tier '{tier}'."

        if not os.path.exists(model_path):
             return f"Error: Model weights missing at {model_path}. Verify 'Elements' SSD."

        cmd = [
            "python3", "-m", "mlx_lm", "generate",
            "--model", model_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temp", "0.7"
        ]

        try:
            print(f"[*] {tier.upper()} Thinking (Async)...")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode()
                return f"Error: {err_msg}"

            return self._clean_output(stdout.decode(), prompt)
        except Exception as e:
            return f"Async Inference Error: {e}"

    def query(self, prompt: str, tier: str = "deep_apex", category: str = "unified", model_type: Optional[str] = None, max_tokens: int = 1000) -> str:
        """
        Execute a local MLX or GGUF query synchronously.
        """
        final_category = model_type if model_type else category
        model_path = self.get_model_path(final_category, tier)
        if not model_path:
            return f"Error: Model category '{final_category}' not found in tier '{tier}'."

        if not os.path.exists(model_path):
            return f"Error: Local model directory missing: {model_path}. Verify 'Elements' SSD."

        print(f"[*] AI Scientist ({tier}/{final_category}) is thinking...")

        cmd = [
            "python3", "-m", "mlx_lm", "generate",
            "--model", model_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temp", "0.7"
        ]

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = time.time()
            latency = end_time - start_time
            print(f"[+] {tier.upper()} Response synthesized in {latency:.2f}s")
            return self._clean_output(result.stdout, prompt)
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"

    async def route_task(self, task_description: str) -> dict:
        """
        Triple-Tier Routing using Google FunctionGemma-270m.
        Returns JSON with tier (deep_apex, regular_scientific, ultra_light_fast).
        """
        prompt = f"""
        Role: PhD AI Orchestrator
        Task: {task_description}
        Objective: Decide which Model Tier and Category should handle this.

        Tiers:
        - deep_apex: High-complexity research, 14B-30B specialists, multi-step formal logic.
        - regular_scientific: Balanced PhD tasks, 8B specialists, standard coding/writing.
        - ultra_light_fast: Edge verification, <4B models, lightning-fast checks.

        Output format: JSON only.
        {{
          "tier": "deep_apex" | "regular_scientific" | "ultra_light_fast",
          "category": "reasoning" | "coding" | "vision" | "logic" | "unified",
          "justification": "short reason"
        }}
        """
        resp = await self.query_async(prompt, tier="ultra_light_fast", category="routing", max_tokens=200)
        try:
            start = resp.find('{')
            end = resp.rfind('}') + 1
            return json.loads(resp[start:end])
        except:
            return {"tier": "regular_scientific", "category": "unified", "justification": "Fallback to Regular Tier"}

    def _clean_output(self, raw: str, prompt: str) -> str:
        ignored = ["deprecated", "RuntimeWarning", "Prompt:", "Generation:", "Peak memory:"]
        if "==========" in raw:
            parts = [p.strip() for p in raw.split("==========") if p.strip()]
            for part in parts:
                if not any(phrase in part for phrase in ignored) and part.strip() != prompt.strip():
                    return part
        return raw.replace(prompt, "").strip()

if __name__ == "__main__":
    orchestrator = ModelOrchestrator()
