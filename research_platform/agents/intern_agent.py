import sys
import os

# Align path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research_platform.model_orchestrator import ModelOrchestrator

class InternAgent:
    """
    Precision Execution & Complex Modeling Agent.
    Specializes in translating high-level ideas into rigorous technical models and execution plans.
    Optimized for Qwen3-30B reasoning and Llama-4-Scout formalization.
    """

    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator
        self.personality = "Precision System & Mathematical Architect"

    async def architect_model(self, blueprint_spec: str, tier: str = "deep_apex"):
        """
        Translates a high-level proposal into a formal technical/mathematical architecture and execution plan.
        """
        prompt = f"""
        Role: {self.personality}
        Proposal: {blueprint_spec}

        Task: Design the formal architecture or logical framework.
        Requirements:
        1. Formalization: Define the core equations, logic gates, or structural blocks.
        2. Efficiency: Optimize for state-of-the-art computational hardware (e.g., M3 Ultra, H100).
        3. Stability: Address convergence, consistency, or structural integrity.
        Output: Full technical specification with formal logic, pseudocode, or class definitions.
        """
        # Use logic category for mathematical ARCHITECTURE, otherwise fallback to reasoning/unified
        category = "logic" if tier == "deep_apex" else "unified"
        return await self.orchestrator.query_async(prompt, tier=tier, category=category, max_tokens=2500)

    def execute_simulation(self, params: str):
        """
        Drafts a Slurm-ready execution script for high-performance clusters.
        """
        prompt = f"""
        Role: {self.personality}
        Parameters: {params}
        Task: Draft a high-performance cluster submission script (Slurm + Python wrapper).
        Optimize for: Multi-node scaling and efficient checkpointing.
        """
        return self.orchestrator.query(prompt, model_type="reasoning", max_tokens=1200)

if __name__ == "__main__":
    orch = ModelOrchestrator()
    intern = InternAgent(orch)
    # print(intern.architect_model("33-track Epigenome-SSM with Test-Time Adaptation"))
