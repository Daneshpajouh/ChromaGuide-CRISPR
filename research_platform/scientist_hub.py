import sys
import os

# Align path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research_platform.model_orchestrator import ModelOrchestrator
from research_platform.agents.scholar import ScholarAgent
from research_platform.agents.scientist import ScientistAgent

class ScientistHub:
    """
    Main Orchestrator for the AI PhD Scientist Platform.
    Coordinates specialized agents for end-to-end research automation.
    """

    def __init__(self):
        print("🚀 Initializing AI PhD Scientist Platform (SOTA Dec 2025)...")
        self.orchestrator = ModelOrchestrator()
        self.scholar = ScholarAgent(self.orchestrator)
        self.scientist = ScientistAgent(self.orchestrator)

    def run_investigation(self, topic: str):
        """
        End-to-end research workflow.
        """
        print(f"\n--- PHASE 1: Literature Synthesis (Scholar) ---")
        lit_review = self.scholar.synthesize_literature(topic)
        print(lit_review)

        print(f"\n--- PHASE 2: Hypothesis Generation (Scientist) ---")
        hypotheses = self.scientist.generate_hypotheses(f"Objective: {topic}\nContext: {lit_review}")
        print(hypotheses)

        # Save output
        timestamp = int(os.time.time()) if hasattr(os, 'time') else "final"
        save_path = f"research_platform/output/investigation_{timestamp}.md"
        with open(save_path, "w") as f:
            f.write(f"# Research Investigation: {topic}\n\n")
            f.write(f"## Literature Review\n{lit_review}\n\n")
            f.write(f"## Generated Hypotheses\n{hypotheses}\n")

        print(f"\n[+] Investigation complete. Results saved to {save_path}")
        return save_path

if __name__ == "__main__":
    hub = ScientistHub()
    # hub.run_investigation("Synergy of Mamba-2 and Geometric Biothermodynamics in CRISPR Prediction")
