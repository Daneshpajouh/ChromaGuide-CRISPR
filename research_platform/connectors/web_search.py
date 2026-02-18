import asyncio
from typing import List, Dict, Optional
from duckduckgo_search import DDGS
from .base import BaseSourceConnector, SourceResult
from datetime import datetime

class WebSearchConnector(BaseSourceConnector):
    """
    Edison v4.0 Global Web Search Connector.
    Uses DuckDuckGo for real-time internet context beyond academic vectors.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key)
        self.source_id = "web"
        self.source_name = "Global Web"

    async def search(self, query: str, limit: int = 20) -> List[SourceResult]:
        """Performs a global web search using DuckDuckGo."""
        self._check_rate_limit()
        print(f"🌐 Searching the global web for: '{query}'...")

        results = []
        try:
            # DDGS is synchronous but we can run it in a thread or just call it if it's fast
            # Modern DDGS supports async as well but let's use the simplest robust way
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=limit))

                for i, res in enumerate(ddg_results):
                    results.append(SourceResult(
                        source_id=self.source_id,
                        source_name=self.source_name,
                        id=f"web-{i}",
                        title=res.get("title", "Untitled"),
                        url=res.get("href", ""),
                        description=res.get("body", ""),
                        content_preview=res.get("body", "")[:500],
                        fetched_at=datetime.now().isoformat(),
                        source_type="documentation"
                    ))
        except Exception as e:
            print(f"Error during web search: {e}")

        return results

    async def fetch_full_content(self, result: SourceResult) -> str:
        """For web results, the snippet is often enough, or we could scrape the URL."""
        return result.description
