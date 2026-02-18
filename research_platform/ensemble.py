import json
import time
import asyncio
import re
from typing import List, Dict, Any
from .model_orchestrator import ModelOrchestrator

class LLMEnsemble:
    """
    Edison v4.0 Local LLM Ensemble.
    Coordinates consensus-based reasoning across multiple LLM tiers.
    """

    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator
        self.last_metrics = {
            "agreement_score": None,
            "disputed_points": [],
            "consensus_at": None
        }

    async def consensus_query(self, prompt: str, category: str = "unified") -> str:
        """
        Runs an ASYNC query through multiple tiers and performs a 'Consensus Audit'.
        True Parallelism: Deep Apex and Turbo Link run simultaneously.
        """
        # Step 1: Parallel Inference
        print("⚡ ENSEMBLE: Launching Parallel Consensus Streams...")
        prime_future = self.orchestrator.query_async(prompt, tier="deep_apex", category=category)
        cross_future = self.orchestrator.query_async(prompt, tier="ultra_light_fast", category=category)

        # Await both simultaneously (cut latency by ~50%)
        prime_res, cross_res = await asyncio.gather(prime_future, cross_future)

        # Step 2: Quantitative Scoring
        scoring_prompt = f"""
        Analyze these two research findings and output a JSON with:
        {{
            "agreement_score": float (0.0 to 1.0),
            "disputed_points": ["list of strings"]
        }}

        Response A: {prime_res[:1000]}
        Response B: {cross_res[:1000]}
        """

        try:
            # Also async
            raw_scores = await self.orchestrator.query_async(scoring_prompt, tier="ultra_light_fast", category="unified")
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', raw_scores, re.DOTALL)
            if json_match:
                metrics = json.loads(json_match.group(0))
                self.last_metrics["agreement_score"] = float(metrics.get("agreement_score", 0.5))
                self.last_metrics["disputed_points"] = metrics.get("disputed_points", [])
                self.last_metrics["consensus_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            self.last_metrics["agreement_score"] = 0.5 # Fallback

        # Step 3: Consensus Refinement
        refinement_prompt = f"""
        Role: Edison Consensus Arbiter (PhD Level)
        Objective: Synthesize two independent research outputs into a single, high-fidelity verified response.

        Task:
        1. Identify any contradictions between the two outputs.
        2. Merge the strengths of both.
        3. Correct any factual errors found in either.
        4. Produce the final authoritative PhD-level response.

        Output 1 (Primary):
        {prime_res[:1500]}

        Output 2 (Cross-Check):
        {cross_res[:1500]}

        Final Authoritative Response:
        """

        final_res = await self.orchestrator.query_async(refinement_prompt, tier="deep_apex", category="unified")
        return final_res

    def get_metrics(self) -> Dict[str, Any]:
        return self.last_metrics
