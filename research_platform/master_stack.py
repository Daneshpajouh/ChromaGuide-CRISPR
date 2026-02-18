import sys
import os
import time
import asyncio
import json

# Align path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research_platform.model_orchestrator import ModelOrchestrator
from research_platform.agents.sakana_v2 import SakanaScientistV2
from research_platform.agents.intern_agent import InternAgent
from research_platform.agents.domain_expert import DomainExpert
from research_platform.agents.coder import CoderAgent
from research_platform.knowledge_retriever import KnowledgeRetriever
from research_platform.verification.proof_checker import ProofCheckingAgent
from research_platform.execution.sandbox import LiveSandbox
from research_platform.verification.hypothesis_engine import HypothesisEvolutionEngine
from research_platform.graph.knowledge_graph import ResearchKnowledgeGraph
from research_platform.graph.graph_miner import GraphMiningAgent
from research_platform.ensemble import LLMEnsemble

class ResearchMaster:
    """
    The Master Gateway for PhD-level autonomous research (Research Hub v14.1).
    Orchestrates Triple-Tier agents (Fast <4B, Regular ~8B, Deep 14B-30B).
    """

    def __init__(self):
        print("🏛️ Initializing Master PhD Scientist Stack (Research Hub v14.1)...")
        self.orchestrator = ModelOrchestrator()
        self.kr = KnowledgeRetriever()
        self.sakana = SakanaScientistV2(self.orchestrator)
        self.intern = InternAgent(self.orchestrator)
        self.expert = DomainExpert(self.orchestrator)
        self.coder = CoderAgent(self.orchestrator)
        self.checker = ProofCheckingAgent(self.orchestrator)
        self.sandbox = LiveSandbox()
        self.hypothesis_engine = HypothesisEvolutionEngine()
        self.graph = ResearchKnowledgeGraph()
        self.miner = GraphMiningAgent(self.orchestrator, self.graph)
        self.ensemble = LLMEnsemble(self.orchestrator)

    async def investigate(self, topic: str, tier: str = "deep_apex", context_str: str = "", log_callback=None, progress_callback=None, task_callback=None) -> str:
        """
        The Master Research Directive: Orchestrates dynamic planning and multi-agent execution.
        SOTA v12.0: Optimized for Mac Studio M3 Ultra + Elements SSD.
        """
        if log_callback: log_callback(f"THINKING: Initializing Research Hub v14.1 Triple-Tier Orchestrator (Tier: {tier.upper()})")

        # Phase 0: Knowledge Aggregation
        print(f"\n🚀 Phase 0: Knowledge Retrieval...")
        source_data = await self.kr.fetch_academic_context(topic, log_callback=log_callback, limit=10)

        # Extract structured sources
        sources = self.kr.aggregator.last_results if hasattr(self.kr.aggregator, 'last_results') else []
        verified_sources_md = "\n".join([f"- **{s.title}** ([{s.source_name}]({s.url}))" for s in sources[:5]])

        # Phase 0.5: Dynamic Planning (Routing via Google FunctionGemma-270m)
        if log_callback: log_callback(f"THINKING: Architecting Triple-Tier Strategy for: '{topic}'...")

        # Intelligent Tier Auto-Selection if 'auto' is passed (Not implemented yet, but we'll stick to user tier)

        planning_prompt = f"""
        Objective: {topic}
        Current Sources: {verified_sources_md}
        Task: You are the Research Hub Master Orchestrator. Architect a high-level research strategy to solve the objective.
        Decide which departments (Ideation, Architecture, Audit, Coding) are required and in what order.

        Output ONLY a JSON list of steps.
        """
        plan_str = await self.orchestrator.query_async(planning_prompt, tier=tier, category="unified", max_tokens=500)
        try:
            plan = json.loads(plan_str[plan_str.find('['):plan_str.rfind(']')+1])
        except:
            plan = [
                {"step": "IDEATION", "goal": f"Initial hypothesis generation for {topic}"},
                {"step": "ARCHITECTURE", "goal": "Technical formalization"},
                {"step": "AUDIT", "goal": "Logic and feasibility audit"},
                {"step": "CODING", "goal": "Implementation blueprint"}
            ]

        if log_callback: log_callback(f"PLANNING: Strategy synthesized with {len(plan)} dynamic milestones.")

        results = {"Topic": topic, "Steps": []}
        current_data = source_data

        # Execute Plan Iteratively
        for i, task in enumerate(plan):
            step_type = task['step']
            goal = task['goal']
            progress = 15 + int((i / len(plan)) * 80)
            if progress_callback: progress_callback(progress)
            if task_callback: task_callback(f"{step_type}: {goal[:30]}...")

            if log_callback: log_callback(f"PHASE {i+1}/{len(plan)}: {step_type} - {goal}")
            if log_callback: log_callback(f"EVENT:STEP_START:{step_type}")

            res = ""
            summary_title = ""
            agent_name = ""

            # Standardized thought tier (Always sub-2B for maximum speed)
            thought_tier = "ultra_light_fast"

            if step_type == "IDEATION":
                agent_name = "ENSEMBLE"
                if log_callback: log_callback(f"EVENT:MODEL_ACTIVE:{agent_name}")
                if log_callback: log_callback(f"EVENT:THOUGHT_START:{agent_name}")
                thought_prompt = f"Role: PhD Scientist. Task: Think deeply about the ideation goal: {goal}. Provide a chain-of-thought analysis."
                thoughts = await self.orchestrator.query_async(thought_prompt, tier=thought_tier, category="unified", max_tokens=500)
                if log_callback: log_callback(f"EVENT:THOUGHT:{thoughts}")
                if log_callback: log_callback(f"EVENT:THOUGHT_END:{agent_name}")

                if tier == "deep_apex":
                    res = await self.ensemble.consensus_query(f"Identify SOTA breakthrough for: {goal} using context: {current_data[:2000]}")
                else:
                    res = await self.sakana.discover(current_data, goal, tier=tier)
                summary_title = "🧠 Process: Ideation Phase Logic & Discovery"
                h_id = self.hypothesis_engine.propose(goal, "Generated during automated ideation phase.")
                if log_callback: log_callback(f"EVENT:HYPOTHESIS_PROPOSED:{h_id}")
            elif step_type == "ARCHITECTURE":
                agent_name = "INTERN"
                if log_callback: log_callback(f"EVENT:MODEL_ACTIVE:{agent_name}")
                if log_callback: log_callback(f"EVENT:THOUGHT_START:{agent_name}")
                thoughts = await self.orchestrator.query_async(f"Think about architecture: {goal}", tier=thought_tier, category="unified", max_tokens=400)
                if log_callback: log_callback(f"EVENT:THOUGHT:{thoughts}")
                if log_callback: log_callback(f"EVENT:THOUGHT_END:{agent_name}")

                if tier == "deep_apex" and "ensemble" in context_str:
                    res = await self.ensemble.consensus_query(f"Architect a robust model structure for: {goal} using context: {current_data[:2000]}")
                else:
                    res = await self.intern.architect_model(current_data, tier=tier)
                summary_title = "📐 Process: Architectural Formalization & Logic"
            elif step_type == "AUDIT":
                agent_name = "EXPERT"
                if log_callback: log_callback(f"EVENT:MODEL_ACTIVE:{agent_name}")
                if log_callback: log_callback(f"EVENT:THOUGHT_START:{agent_name}")
                thoughts = await self.orchestrator.query_async(f"Think about safety audit: {goal}", tier=thought_tier, category="unified", max_tokens=400)
                if log_callback: log_callback(f"EVENT:THOUGHT:{thoughts}")
                if log_callback: log_callback(f"EVENT:THOUGHT_END:{agent_name}")

                res = await self.expert.safety_audit(current_data, tier=tier)
                summary_title = "🛡️ Process: Red-Team Risk Assessment & Domain Audit"
            elif step_type == "CODING":
                agent_name = "CODER"
                if log_callback: log_callback(f"EVENT:MODEL_ACTIVE:{agent_name}")
                if log_callback: log_callback(f"EVENT:THOUGHT_START:{agent_name}")
                thoughts = await self.orchestrator.query_async(f"Think about implementation: {goal}", tier=thought_tier, category="unified", max_tokens=400)
                if log_callback: log_callback(f"EVENT:THOUGHT:{thoughts}")
                if log_callback: log_callback(f"EVENT:THOUGHT_END:{agent_name}")

                res = await self.coder.refactor_code("blueprint.py", current_data, tier=tier)
                summary_title = "💻 Process: Technical Blueprinting & Implementation"

            # Proof-Checking
            audit_data = await self.checker.audit_step(step_type, goal, res, tier=tier)
            audit_report = self.checker.format_audit_report(audit_data)

            # Sandbox
            sandbox_report = ""
            if step_type == "CODING" and "```python" in res:
                code_snippet = res[res.find("```python")+9:res.find("```", res.find("```python")+9)].strip()
                exec_result = self.sandbox.execute(code_snippet)
                sandbox_report = f"\n### 🧪 Sandbox Execution [{exec_result['status'].upper()}]\n```\n{exec_result['output'][:500]}\n```\n"

            results["Steps"].append({"phase": i+1, "type": step_type, "goal": goal, "output": f"<details>\n<summary>{summary_title}</summary>\n\n{res}\n\n{audit_report}\n{sandbox_report}\n</details>"})

            # Distill findings for next phase
            distillation_prompt = f"Distill the following research output into a concise PhD-level summary of findings for the next phase: {res[:2000]}"
            summary_findings = await self.orchestrator.query_async(distillation_prompt, tier=thought_tier, category="unified", max_tokens=300)
            current_data = f"Background Context: {source_data[:1000]}\n\nPrevious Findings: {summary_findings}"

        # Final Report
        report_path = f"research_platform/output/master_investigation_{int(time.time())}.md"
        with open(report_path, "w") as f:
            f.write(f"# 🧪 Research Hub v14.1 Triple-Tier Strategic Research Report: {topic}\n")
            f.write(f"> **Orchestration Tier:** {tier.upper()} | **Storage:** Elements SSD\n\n")
            for step in results["Steps"]:
                f.write(f"## Phase {step['phase']}: {step['type']}\n{step['output']}\n\n---\n\n")

        return report_path

if __name__ == "__main__":
    master = ResearchMaster()
