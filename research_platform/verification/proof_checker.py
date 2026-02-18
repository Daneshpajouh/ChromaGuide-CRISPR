from typing import Dict, Any
import json
from ..model_orchestrator import ModelOrchestrator

class ProofCheckingAgent:
    """
    Edison v4.0 Proof-Checking Engine.
    Performs parallel logic audits and red-team verification of research steps.
    """

    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator

    async def audit_step(self, step_type: str, step_goal: str, step_output: str, tier: str = "deep_apex") -> Dict[str, Any]:
        """
        Performs a rigorous audit of a research step's output.
        """
        audit_prompt = f"""
        Role: Edison Senior Research Auditor (PhD-Level)
        Objective: Rigorously audit the logical integrity of a research task.

        Task Type: {step_type}
        Goal: {step_goal}
        Output to Audit:
        ---
        {step_output[:3000]}
        ---

        Audit Requirements:
        1. Logical Consistency: Identify any contradictions or leaps in reasoning.
        2. Factual Integrity: Highlight potential hallucinations or unverified technical claims.
        3. Structural Rigor: Evaluate if the architecture/code follows SOTA best practices.
        4. Safety & Ethics: Flag any dual-use or biological safety risks.

        Output ONLY a JSON object with this structure:
        {{
            "integrity_score": <0.0 to 1.0>,
            "critical_flaws": ["List of major issues"],
            "minor_optimizations": ["List of improvements"],
            "verification_status": "PASSED" | "FAILED",
            "audit_summary": "Brief summary of the audit findings"
        }}
        """

        res = await self.orchestrator.query_async(audit_prompt, tier=tier, category="unified", max_tokens=800)

        # Robust JSON extraction
        import re
        try:
            # Find the largest {...} block
            json_match = re.search(r'\{.*\}', res, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                raw_data = json.loads(json_str)

                # Enforce schema and defaults to prevent KeyErrors
                audit_data = {
                    "integrity_score": float(raw_data.get("integrity_score", 0.5)),
                    "critical_flaws": raw_data.get("critical_flaws", []),
                    "minor_optimizations": raw_data.get("minor_optimizations", []),
                    "verification_status": raw_data.get("verification_status", "PASSED"),
                    "audit_summary": raw_data.get("audit_summary", "No summary provided by auditor.")
                }

                # Validation: ensure flaws/optimizations are lists
                if not isinstance(audit_data["critical_flaws"], list):
                    audit_data["critical_flaws"] = [str(audit_data["critical_flaws"])]
                if not isinstance(audit_data["minor_optimizations"], list):
                    audit_data["minor_optimizations"] = [str(audit_data["minor_optimizations"])]

                return audit_data
            else:
                raise ValueError("No JSON block found in response")
        except Exception as e:
            # Attempt to extract fields from raw text if JSON fails
            return {
                "integrity_score": 0.5 if "passed" in res.lower() else 0.2,
                "critical_flaws": [f"Audit Engine Parsing Warning: {str(e)}"],
                "minor_optimizations": ["Ensure model outputs valid JSON in future steps"],
                "verification_status": "PASSED" if "passed" in res.lower() else "FAILED",
                "audit_summary": f"Partial Audit (Raw Text Analysis): {res[:200]}..."
            }

    def format_audit_report(self, audit_data: Dict[str, Any]) -> str:
        """Formats the audit data into a professional markdown block."""
        status_emoji = "✅" if audit_data['verification_status'] == "PASSED" else "❌"
        score = int(audit_data['integrity_score'] * 100)

        report = f"### 🛡️ Audit Report: [{audit_data['verification_status']}] (Integrity: {score}%)\n"
        report += f"**Summary:** {audit_data['audit_summary']}\n\n"

        if audit_data['critical_flaws']:
            report += "**Critical Flaws:**\n"
            for flaw in audit_data['critical_flaws']:
                report += f"- {flaw}\n"

        if audit_data['minor_optimizations']:
            report += "\n**Suggested Optimizations:**\n"
            for opt in audit_data['minor_optimizations']:
                report += f"- {opt}\n"

        return report
