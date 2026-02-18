import sys
import os

# Align path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research_platform.model_orchestrator import ModelOrchestrator

class ScientistAgent:
    """
    Bio-Genomics Research Scientist Agent.
    Specializes in hypothesis generation, experimental design, and biophysical modeling.
    Uses 'Thinking' models for high reasoning density.
    """

    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator
        self.role = "Senior Research Scientist (CRISPR & Biophysics)"

    def generate_hypotheses(self, problem_statement: str):
        """
        Produce novel scientific hypotheses for a given challenge.
        """
        prompt = f"""
        Role: {self.role}
        Problem: {problem_statement}
        Task: Utilize 'Thinking' mode to iterate through 5 candidate hypotheses.
        Focus: Geometric Biothermodynamics and Mamba-2 sequences.
        Output: For each hypothesis:
        - [Mechanism]: The underlying biophysical process.
        - [SOTA Gap]: Why current methods (e.g. DeepMEns) miss this.
        - [Validation]: How we would test this on the Nibi cluster.
        """
        return self.orchestrator.query(prompt, model_type="thinking", max_tokens=2500)

    def design_experiment(self, hypothesis: str):
        """
        Translate a hypothesis into a concrete experimental script or configuration.
        """
        prompt = f"""
        Role: {self.role}
        Hypothesis: {hypothesis}
        Task: Draft the pseudocode/architecture for a new Mamba-2 layer that tests this.
        Focus: Efficiency, Gradient Flow, and Information Density.
        """
        return self.orchestrator.query(prompt, model_type="thinking", max_tokens=2000)

if __name__ == "__main__":
    # Test
    orch = ModelOrchestrator()
    scientist = ScientistAgent(orch)
    print(scientist.generate_hypotheses("Improving SaCas9 few-shot transfer learning"))
