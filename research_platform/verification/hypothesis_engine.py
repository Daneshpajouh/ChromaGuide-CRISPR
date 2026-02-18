from typing import List, Dict, Any
from dataclasses import dataclass, field
import datetime

@dataclass
class Hypothesis:
    id: str
    content: str
    rationale: str
    confidence_score: float = 0.5
    status: str = "proposed" # proposed, verified, rejected, evolved
    evidence: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

class HypothesisEvolutionEngine:
    """
    Edison v4.0 Hypothesis Evolution Engine.
    Tracks the lifecycle of scientific hypotheses and documents their evolution.
    """

    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.evolution_log: List[str] = []

    def propose(self, content: str, rationale: str) -> str:
        h_id = f"H-{len(self.hypotheses) + 1:03d}"
        self.hypotheses[h_id] = Hypothesis(id=h_id, content=content, rationale=rationale)
        self.evolution_log.append(f"[{datetime.datetime.now().isoformat()}] PROPOSED: {h_id} - {content[:50]}...")
        return h_id

    def update_evidence(self, h_id: str, evidence: str, impact: float):
        if h_id not in self.hypotheses: return

        hyp = self.hypotheses[h_id]
        hyp.evidence.append(evidence)
        hyp.confidence_score = max(0.0, min(1.0, hyp.confidence_score + impact))

        if hyp.confidence_score > 0.8:
            hyp.status = "verified"
        elif hyp.confidence_score < 0.2:
            hyp.status = "rejected"

        self.evolution_log.append(f"[{datetime.datetime.now().isoformat()}] EVIDENCE: {h_id} - Confidence now {hyp.confidence_score:.2f}")

    def evolve(self, h_id: str, new_content: str, reason: str) -> str:
        if h_id not in self.hypotheses: return ""

        old_hyp = self.hypotheses[h_id]
        old_hyp.status = "evolved"

        new_id = f"{h_id}.v{len([k for k in self.hypotheses if k.startswith(h_id)]) + 1}"
        self.hypotheses[new_id] = Hypothesis(
            id=new_id,
            content=new_content,
            rationale=f"Evolved from {h_id} because: {reason}",
            confidence_score=old_hyp.confidence_score
        )

        self.evolution_log.append(f"[{datetime.datetime.now().isoformat()}] EVOLVED: {h_id} -> {new_id}")
        return new_id

    def generate_research_git_log(self) -> str:
        """Generates a 'Research Git' style log of all hypothesis changes."""
        log = "## 📜 Research Git: Hypothesis Evolution Log\n\n"
        for entry in self.evolution_log:
            log += f"- {entry}\n"
        return log

    def summarize_hypotheses(self) -> str:
        summary = "## 🧬 Hypotheses Status\n\n"
        for h_id, hyp in self.hypotheses.items():
            status_color = "🟢" if hyp.status == "verified" else "🔴" if hyp.status == "rejected" else "🟡"
            summary += f"### {status_color} {h_id}: {hyp.status.upper()}\n"
            summary += f"**Content:** {hyp.content}\n"
            summary += f"**Confidence:** {hyp.confidence_score:.2f}\n"
            summary += f"**Rationale:** {hyp.rationale}\n\n"
        return summary
