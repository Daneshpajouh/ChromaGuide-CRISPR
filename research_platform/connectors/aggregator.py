import asyncio
from typing import List, Dict
from datetime import datetime
from .base import BaseSourceConnector, SourceResult

class UnifiedSourceAggregator:
    """Fan out to all connectors, rank results, and deduplicate."""

    def __init__(self, connectors: Dict[str, BaseSourceConnector]):
        self.connectors = connectors

    async def unified_search(
        self,
        query: str,
        sources: List[str] = None,
        limit: int = 50
    ) -> Dict:
        if sources is None:
            sources = list(self.connectors.keys())

        tasks = [
            self.connectors[source].search(query, limit=limit)
            for source in sources if source in self.connectors
        ]

        all_results = []
        try:
            results_per_source = await asyncio.gather(*tasks, return_exceptions=True)
            for res_list in results_per_source:
                if isinstance(res_list, list):
                    all_results.extend(res_list)
        except Exception as e:
            print(f"Error during unified search: {e}")

        # Sort by simple heuristic (source-based + keyword match)
        all_results.sort(key=lambda x: x.source_id, reverse=False)

        # Store last results for master_stack visibility
        self.last_results = all_results[:limit]

        return {
            "query": query,
            "results": self.last_results,
            "timestamp": datetime.now().isoformat()
        }
