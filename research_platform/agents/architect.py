import sys
import os

# Align path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research_platform.model_orchestrator import ModelOrchestrator

class ArchitectAgent:
    """
    Bio-Genomics Manuscript Architect Agent.
    Specializes in drafting Nature-style papers, sections, and LaTeX formatting.
    """

    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator
        self.role = "Senior Scientific Architect & Editor (Nature Methods)"

    def draft_section(self, section_name: str, data: str, context: str):
        """
        Draft a specific section of a scientific paper.
        """
        prompt = f"""
        Role: {self.role}
        Section: {section_name}
        Experimental Data: {data}
        Project Context: {context}

        Task: Draft a high-impact, professional {section_name} for a Nature Methods submission.
        Instructions:
        1. Use active voice and precise terminology.
        2. Ensure the narrative links the biophysical mechanisms to the SOTA improvement.
        3. Format using standard scientific conventions.
        """
        return self.orchestrator.query(prompt, model_type="reasoning", max_tokens=2000)

    def refine_abstract(self, current_abstract: str):
        """
        Iteratively refine a research abstract for maximum impact.
        """
        prompt = f"""
        Role: {self.role}
        Current Abstract: {current_abstract}
        Task: Refine this for 'High Impact' and 'Clarity'.
        Requirement: Must emphasize the Rho > 0.88 achievement and Geometric Biothermodynamics novelty.
        """
        return self.orchestrator.query(prompt, model_type="reasoning", max_tokens=800)

if __name__ == "__main__":
    # Test
    orch = ModelOrchestrator()
    architect = ArchitectAgent(orch)
    print(architect.refine_abstract("We built a Mamba model for CRISPR. It works well."))
