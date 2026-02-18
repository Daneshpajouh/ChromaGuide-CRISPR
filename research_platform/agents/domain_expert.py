import sys
import os

# Align path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research_platform.model_orchestrator import ModelOrchestrator

class DomainExpert:
    """
    Topic-Specific Specialist Auditor.
    Focuses on domain precision, risk assessment, and technical viability across any PhD field.
    Optimized for Qwen3-30B-A3B and InternVL3.5 multimodal analysis.
    """

    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator
        self.personality = "Lead SOTA Auditor & Domain Specialist"

    async def safety_audit(self, technical_spec: str, tier: str = "deep_apex"):
        """
        Perform a rigorous safety, alignment, and 'Red Team' audit of a proposal.
        """
        prompt = f"""
        Role: {self.personality}
        Technical Specification: {technical_spec}
        Task: Perform a 'Red Team' audit of this architecture/proposal.
        Analyze:
        1. Failure Modes: Identify edge cases where the system crashes or yields invalid results.
        2. Security & Alignment: Assess risks related to data privacy, domain-specific safety, or AI alignment.
        3. Scalability Bottlenecks: Highlight where the design fails under extreme load.
        """
        category = "reasoning" if tier == "deep_apex" else "unified"
        return await self.orchestrator.query_async(prompt, tier=tier, category=category, max_tokens=1500)

    def optimize_parameters(self, target: str, objective: str):
        """
        Design the optimal parameter strategy or intervention for a specific goal.
        """
        prompt = f"""
        Role: {self.personality}
        Target: {target}
        Objective: {objective}
        Task: Design a multi-variable optimization strategy to achieve this with 99.9% certainty.
        Constraint: Minimize unwanted side-effects and maximize efficiency of the intervention.
        """
        return self.orchestrator.query(prompt, model_type="reasoning", max_tokens=2000)

if __name__ == "__main__":
    orch = ModelOrchestrator()
    expert = DomainExpert(orch)
    # print(expert.optimize_parameters("Fluid Dynamics Simulation", "Minimize turbulence in hypersonic conditions"))
