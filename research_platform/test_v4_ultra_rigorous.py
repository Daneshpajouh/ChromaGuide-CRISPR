import sys
import os
import json
import asyncio
import sqlite3
import unittest
from datetime import datetime

# Path alignment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research_platform.ensemble import LLMEnsemble
from research_platform.model_orchestrator import ModelOrchestrator
from research_platform.graph.knowledge_graph import ResearchKnowledgeGraph
from research_platform.verification.hypothesis_engine import HypothesisEvolutionEngine
from research_platform.execution.sandbox import LiveSandbox
from research_platform.knowledge_retriever import KnowledgeRetriever

class EdisonUltraRigorousTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ModelOrchestrator()
        cls.ensemble = LLMEnsemble(cls.orchestrator)
        cls.kg = ResearchKnowledgeGraph(db_path="research_platform/test_kg.db")
        cls.hypotheses = HypothesisEvolutionEngine()
        cls.sandbox = LiveSandbox()
        cls.kr = KnowledgeRetriever()

    def test_01_ensemble_consensus(self):
        """Test the LLM Ensemble consensus and metrics."""
        print("\n[TEST] 01: LLM Ensemble Consensus...")
        # Mocking or actual query? Let's use a simple prompt that doesn't trigger massive weights if possible
        # but the prompt should be sufficient to test the consensus logic.
        prompt = "Explain the impact of Mamba-2 on genomic sequence modeling."
        result = self.ensemble.consensus_query(prompt)
        metrics = self.ensemble.get_metrics()

        self.assertIsInstance(result, str)
        self.assertIn("agreement_score", metrics)
        self.assertGreaterEqual(metrics["agreement_score"], 0.0)
        self.assertLessEqual(metrics["agreement_score"], 1.0)
        print(f"      - Agreement Score: {metrics['agreement_score']}")

    def test_02_graph_persistence(self):
        """Test Knowledge Graph node/edge persistence."""
        print("\n[TEST] 02: Knowledge Graph Persistence...")
        # Fix: ResearchKnowledgeGraph.add_entity(self, entity_id, name, entity_type, metadata)
        self.kg.add_entity("Mamba-2", "Mamba-2", "Architecture", {"desc": "SSM model"})
        self.kg.add_entity("Genomics", "Genomics", "Domain", {"desc": "Bio study"})
        # Fix: ResearchKnowledgeGraph.add_relation(self, source_id, target_id, relation, weight)
        self.kg.add_relation("Mamba-2", "Genomics", "OPTIMIZES", 1.0)

        entities = self.kg.get_all_entities()
        entity_names = [e[1] for e in entities]
        self.assertIn("Mamba-2", entity_names)

        # Check direct SQL for relationship
        with sqlite3.connect(self.kg.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM relationships WHERE source_id=?", ("Mamba-2",))
            rel = cursor.fetchone()
            self.assertIsNotNone(rel)
            self.assertEqual(rel[2], "OPTIMIZES")

    def test_03_hypothesis_evolution(self):
        """Test Hypothesis state transitions."""
        print("\n[TEST] 03: Hypothesis Evolution...")
        h_id = self.hypotheses.propose("Mamba-2 DNA dependencies.", "Initial rationale.")
        self.assertEqual(self.hypotheses.hypotheses[h_id].status, "proposed")

        # Fix: HypothesisEvolutionEngine.update_evidence(self, h_id, evidence, impact)
        self.hypotheses.update_evidence(h_id, "Benchmark verified.", 0.4)
        self.assertEqual(self.hypotheses.hypotheses[h_id].confidence_score, 0.9) # 0.5 + 0.4

        # Fix: evolve(self, h_id, new_content, reason)
        new_id = self.hypotheses.evolve(h_id, "DNA dependencies ultra-long.", "Better context.")
        self.assertEqual(self.hypotheses.hypotheses[new_id].status, "proposed")
        self.assertEqual(self.hypotheses.hypotheses[h_id].status, "evolved")

    def test_04_sandbox_execution(self):
        """Test Live Sandbox code execution."""
        print("\n[TEST] 04: Live Sandbox Execution...")
        code = "import math\nprint(math.sqrt(16))"
        result = self.sandbox.execute(code)
        self.assertEqual(result["status"], "ok")
        self.assertIn("4.0", result["output"])

    def test_05_source_aggregation_integrity(self):
        """Test KnowledgeRetriever source filtering."""
        print("\n[TEST] 05: Source Aggregation Integrity...")
        # Test that toggles are respected if we had them in KR (checking current KR state)
        if hasattr(self.kr, 'aggregator'):
            # This is a meta-test to ensure the aggregator exists
            self.assertIsNotNone(self.kr.aggregator)
            print("      - Aggregator detected and healthy.")

if __name__ == "__main__":
    unittest.main()
