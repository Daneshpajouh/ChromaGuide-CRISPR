import os
import asyncio
import nest_asyncio
from typing import List, Dict
from .connectors.github import GitHubConnector
from .connectors.huggingface import HuggingFaceConnector
from .connectors.arxiv import ArxivConnector
from .connectors.pubmed import PubMedConnector
from .connectors.biorxiv import BioRxivConnector
from .connectors.linkedin import LinkedInConnector
from .connectors.hf_papers import HFPapersConnector
from .connectors.web_search import WebSearchConnector
from .connectors.aggregator import UnifiedSourceAggregator

# Allow nested event loops for Edison v4.0 bridges
nest_asyncio.apply()

class KnowledgeRetriever:
    """
    Connects the PhD Hub to global academic and technological sources.
    Uses the v4.0 Unified Source Aggregator for deep multi-vector retrieval.
    """

    def __init__(self):
        # Initialize connectors
        connectors = {
            "github": GitHubConnector(),
            "huggingface": HuggingFaceConnector(),
            "arxiv": ArxivConnector(),
            "pubmed": PubMedConnector(),
            "biorxiv": BioRxivConnector(),
            "web": WebSearchConnector(),
            "linkedin": LinkedInConnector(),
            "huggingface_daily": HFPapersConnector()
        }
        self.aggregator = UnifiedSourceAggregator(connectors)

    async def fetch_academic_context(self, topic: str, limit: int = 50, log_callback=None) -> str:
        """
        Retrieves real-time summaries from major academic and technical sources asynchronously.
        """
        msg = f"📡 Edison v4.0 Unified Bridge: Architecting deep search strategy for '{topic}'..."
        print(msg)
        if log_callback: log_callback(f"SEARCHING: {msg}")

        # Direct async await (No nested loop hacks needed in v4.0 Async Core)
        results_dict = await self.aggregator.unified_search(topic, limit=limit)

        all_results = results_dict.get("results", [])

        if not all_results:
            return "No specialized SOTA papers or repositories found in the primary vectors."

        output = f"--- Edison v4.0 Consolidated Intelligence ---\n"
        source_counts = {}

        for res in all_results:
            source_counts[res.source_name] = source_counts.get(res.source_name, 0) + 1
            output += f"[{res.source_name}] {res.title} \n"
            output += f"  - URL: {res.url}\n"
            output += f"  - Snippet: {res.content_preview[:200]}...\n\n"

        # Logs for transparency
        for source, count in source_counts.items():
            if log_callback: log_callback(f"FOUND: Integrating {count} specialized items from {source}.")

        msg = f"✅ Success: Consolidated intelligence from {len(all_results)} specialized SOTA vectors."
        print(msg)
        if log_callback: log_callback(f"INTEGRATING: {msg}")

        return output

if __name__ == "__main__":
    kr = KnowledgeRetriever()
    print(kr.fetch_academic_context("Mamba architectures in Genomics"))

if __name__ == "__main__":
    # Test
    kr = KnowledgeRetriever()
    print(kr.fetch_academic_context("Mamba architectures in Genomics"))
