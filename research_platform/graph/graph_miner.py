from typing import List, Dict, Any
import json
from ..model_orchestrator import ModelOrchestrator
from .knowledge_graph import ResearchKnowledgeGraph

class GraphMiningAgent:
    """
    Edison v4.0 Graph-Mining Agent.
    Analyzes research text to extract entities and semantic relationships for the Knowledge Graph.
    """

    def __init__(self, orchestrator: ModelOrchestrator, graph: ResearchKnowledgeGraph):
        self.orchestrator = orchestrator
        self.graph = graph

    def mine_text(self, text: str, tier: str = "deep_apex"):
        """
        Processes research text and populates the Knowledge Graph.
        """
        mining_prompt = f"""
        Role: Semantic Knowledge Extractor (PhD Level)
        Objective: Convert research findings into a structured semantic graph.

        Text to Analyze:
        ---
        {text[:4000]}
        ---

        Task:
        1. Identify key entities (Topics, Models, Papers, Hypotheses, Chemicals, Genes, etc.).
        2. Identify semantic relationships (references, optimizes, contradicts, is-a, regulates, etc.).

        Output ONLY a JSON object with this structure:
        {{
            "entities": [
                {{"id": "E001", "name": "Mamba-2", "type": "Model", "metadata": {{"version": "2.0"}} }}
            ],
            "relations": [
                {{"source": "E001", "target": "TRANSFORMER", "relation": "optimizes"}}
            ]
        }}
        """

        res = self.orchestrator.query(mining_prompt, tier=tier, category="unified", max_tokens=1000)

        try:
            json_start = res.find('{')
            json_end = res.rfind('}') + 1
            mining_data = json.loads(res[json_start:json_end])

            # Populate Graph
            for ent in mining_data.get('entities', []):
                self.graph.add_entity(ent['id'], ent['name'], ent['type'], ent.get('metadata', {}))

            for rel in mining_data.get('relations', []):
                self.graph.add_relation(rel['source'], rel['target'], rel['relation'])

            return len(mining_data.get('entities', [])), len(mining_data.get('relations', []))
        except Exception as e:
            print(f"⚠️ Graph Mining Failed: {e}")
            return 0, 0

    async def mine_text_async(self, text: str, tier: str = "deep_apex"):
        """
        Background Async Mining (Non-Blocking).
        """
        mining_prompt = f"""
        Role: Semantic Knowledge Extractor (PhD Level) (ASYNC)
        Objective: Convert research findings into a structured semantic graph.

        Text to Analyze:
        ---
        {text[:4000]}
        ---

        Task:
        1. Identify key entities (Topics, Models, Papers, Hypotheses, Chemicals, Genes, etc.).
        2. Identify semantic relationships (references, optimizes, contradicts, is-a, regulates, etc.).

        Output ONLY a JSON object with this structure:
        {{
            "entities": [
                {{"id": "E001", "name": "Mamba-2", "type": "Model", "metadata": {{"version": "2.0"}} }}
            ],
            "relations": [
                {{"source": "E001", "target": "TRANSFORMER", "relation": "optimizes"}}
            ]
        }}
        """

        res = await self.orchestrator.query_async(mining_prompt, tier=tier, category="unified", max_tokens=1000)

        try:
            json_start = res.find('{')
            json_end = res.rfind('}') + 1
            mining_data = json.loads(res[json_start:json_end])

            # Populate Graph
            for ent in mining_data.get('entities', []):
                self.graph.add_entity(ent['id'], ent['name'], ent['type'], ent.get('metadata', {}))

            for rel in mining_data.get('relations', []):
                self.graph.add_relation(rel['source'], rel['target'], rel['relation'])

            print(f"✅ Background Graph Mining Complete: {len(mining_data.get('entities', []))} nodes.")
            return len(mining_data.get('entities', [])), len(mining_data.get('relations', []))
        except Exception as e:
            print(f"⚠️ Async Graph Mining Failed: {e}")
            return 0, 0
