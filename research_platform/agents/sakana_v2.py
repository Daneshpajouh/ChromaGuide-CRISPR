import sys
import os

# Align path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research_platform.model_orchestrator import ModelOrchestrator

class SakanaScientistV2:
    """
    Autonomous Scientific Discovery Agent (Sakana AI v2 Logic).
    Specializes in open-ended ideation, evolution of hypotheses, and automated peer review.
    Optimized for Nanbeige4-Thinking at 200+ t/s.
    """

    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator
        self.personality = "Edison-Level Multi-Modal Ideator"

    def discover(self, domain_context: str, objective: str, tier: str = "deep_apex"):
        """
        The 'Ideation Loop': Generates and evolves a novel research/technical proposal.
        """
        prompt = f"""
        Role: {self.personality}
        Background Context: {domain_context}
        Objective: {objective}

        Task: Perform a deep recursive ideation loop.
        1. Contextual Analysis: Expand the objective into a multi-layer hypothesis based on the latest SOTA.
        2. Friction Identification: Identify the primary 'Impossible Constraint' or bottleneck.
        3. Strategic Resolution: Propose a radical solution (mathematical, architectural, or experimental).
        4. Synthesis: Draft a high-impact 'Nature/CVPR-Ready' Technical Proposal.
        """
        return self.orchestrator.query(prompt, tier=tier, category="unified", max_tokens=3000)

    def peer_review(self, manuscript: str):
        """
        Critical Review Protocol: Simulates a top-tier journal reviewer (e.g., Nature, Science, CVPR).
        """
        prompt = f"""
        Role: Senior Editor at a Top-Tier Scientific Journal
        Manuscript: {manuscript}
        Task: Provide a devastatingly rigorous peer review.
        Identify: Methodological weaknesses, missing controls, logical fallacies, and flaws in the formal derivation.
        """
        return self.orchestrator.query(prompt, model_type="reasoning", max_tokens=1500)

if __name__ == "__main__":
    orch = ModelOrchestrator()
    sakana = SakanaScientistV2(orch)
    print(sakana.discover("Computational Genomics", "Non-linear R-loop kinetics in Mamba-2 SSMs"))
