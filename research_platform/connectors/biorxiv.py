import httpx
from typing import List
from .base import BaseSourceConnector, SourceResult

class BioRxivConnector(BaseSourceConnector):
    """Retrieves preprints from bioRxiv API."""

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.biorxiv.org/details/biorxiv"

    async def search(self, query: str, limit: int = 5) -> List[SourceResult]:
        # bioRxiv API is slightly different (usually by date or DOI),
        # for a query, we'll use their search endpoint if available or a mock for this demo
        # Actually bioRxiv doesn't have a simple 'query' API like arXiv, often requires scrapers
        # We will use the arXiv API as a proxy for bioRxiv content if needed,
        # but for now, we'll implement a basic fetch of recent relevant papers.

        results = []
        try:
            # For demonstration, we'll use the 'recent' API
            async with httpx.AsyncClient() as client:
                # bioRxiv uses segments of 100
                res = await client.get(f"{self.base_url}/recent/0/{limit}")
                if res.status_code == 200:
                    data = res.json()
                    for paper in data.get('collection', []):
                        # Filter by simple keyword match since bioRxiv API is limited on Search
                        if query.lower() in paper['title'].lower() or query.lower() in paper['abstract'].lower():
                            results.append(SourceResult(
                                source_id="biorxiv",
                                source_name="bioRxiv",
                                id=paper['doi'],
                                title=paper['title'],
                                url=f"https://www.biorxiv.org/content/{paper['doi']}",
                                description=paper['abstract'][:300],
                                content_preview=paper['abstract'][:500],
                                metadata={"authors": paper['authors']},
                                source_type="paper"
                            ))
        except Exception as e:
            print(f"bioRxiv Search Failed: {e}")

        return results

    async def fetch_full_content(self, result: SourceResult) -> str:
        # bioRxiv doesn't have a simple TXT/Markdown API for full text.
        # We'll return the description (abstract) as a fallback.
        return result.description
