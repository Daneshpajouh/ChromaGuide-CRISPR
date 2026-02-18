import sys
import os

# Align path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research_platform.model_orchestrator import ModelOrchestrator

class ScholarAgent:
    """
    Bio-Genomics Research Scholar Agent.
    Specializes in literature mining, synthesis, and critique of SOTA claims.
    """

    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator
        self.role = "Senior Research Scholar in Computational Genomics"

    def synthesize_literature(self, query: str):
        """
        Produce a synthesis of the current state of a field.
        """
        prompt = f"""
        Role: {self.role}
        Query: {query}
        Task: Synthesize the current Dec 2025 SOTA landscape for this topic.
        Instructions:
        1. Identify the 3 most promising architectures.
        2. Detail the primary performance bottleneck remaining.
        3. Formulate a 'Golden Thread' narrative connecting current results to a future Nobel-level breakthrough.
        """
        return self.orchestrator.query(prompt, model_type="reasoning", max_tokens=1500)

    def run_power_query(self, query_id: int):
        """
        Executes one of the ultra-high-density 'Edison Power Queries'.
        """
        queries = {
            1: "Perform a comprehensive cross-disciplinary synthesis evaluating the intersection of State Space Models (SSMs), specifically Mamba-2/Mamba-3 architectures, with the Geometric Biothermodynamics of CRISPR-Cas9 R-loop kinetics. Analyze the theoretical precedent for replacing traditional discrete BiLSTM/Transformer encoders with topology-aware SSMs to capture the non-linear energetic landscape of DNA supercoiling and chromatin accessibility.",
            2: "Analyze the current state-of-the-art in 3D-Aware Genomic Machine Learning, specifically the integration of Graph Neural Networks (GNNs) like the AttO3D layer. Compare the performance gains of Latent Diffusion Models (LDMs) vs. Autoregressive Transformers in generating functional sgRNAs.",
            3: "Conduct a deep literature audit on the extraction of human-interpretable 'Nucleomer Rules' from deep learning models using Mamba Saliency Mapping and Latent Space Interpolation."
        }
        prompt = queries.get(query_id, "Analyze the CRISPR SOTA landscape.")
        return self.orchestrator.query(prompt, model_type="reasoning", max_tokens=2000)

if __name__ == "__main__":
    # Test
    orch = ModelOrchestrator()
    scholar = ScholarAgent(orch)
    print(scholar.synthesize_literature("Mamba-2 vs Transformers for Genomic Modeling"))
