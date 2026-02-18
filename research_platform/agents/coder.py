import sys
import os

# Align path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research_platform.model_orchestrator import ModelOrchestrator

class CoderAgent:
    """
    Autonomous Coding Agent (Goose/Roo Code Logic).
    Specializes in refactoring, dependency management, and local repo modifications.
    """

    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator
        self.role = "Senior Software Architect & Coder"

    async def refactor_code(self, file_path: str, request: str, tier: str = "deep_apex"):
        """
        Drafts a refactoring plan for a specific local file.
        Uses specialized Qwen3-Coder (30B) for peak agentic performance.
        """
        prompt = f"""
        Role: {self.role}
        File: {file_path}
        Request: {request}
        Task: Draft a precise, diff-ready refactoring plan.
        Focus:
        1. Performance optimization for M3 Ultra.
        2. Modern Python 3.10+ PEP standards.
        3. Scalability and modularity.
        """
        return await self.orchestrator.query_async(prompt, tier=tier, category="coding", max_tokens=2000)

    def audit_repo(self, repo_structure: str):
        """
        Audits the repository for dead code, security risks, or architectural debt.
        """
        prompt = f"""
        Role: {self.role}
        Structure: {repo_structure}
        Task: Perform a full architectural health audit.
        Identify: Spaghetti code, lack of abstraction, and potential data leakage points.
        """
        return self.orchestrator.query(prompt, model_type="reasoning", max_tokens=1500)

if __name__ == "__main__":
    orch = ModelOrchestrator()
    coder = CoderAgent(orch)
    print(coder.refactor_code("src/train_deepmens.py", "Optimize the one-hot encoding for speed."))
