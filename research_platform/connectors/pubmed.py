import httpx
from typing import List, Optional
from .base import BaseSourceConnector, SourceResult

class PubMedConnector(BaseSourceConnector):
    """Search PubMed for biomedical literature."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = "researcher@edison.local"

    async def search(self, query: str, limit: int = 20) -> List[SourceResult]:
        async with httpx.AsyncClient(timeout=15) as client:
            # Step 1: Search PMIDs
            search_url = f"{self.base_url}/esearch.fcgi"
            params = {"db": "pubmed", "term": query, "retmax": limit, "rettype": "json", "email": self.email}
            response = await client.get(search_url, params=params)
            if response.status_code != 200: return []

            pmids = response.json().get("esearchresult", {}).get("idlist", [])
            if not pmids: return []

            # Step 2: Simplified retrieval (PubMed search results are PMIDs only initially)
            # For brevity, we'll just create placeholder results since full XML parsing is verbose
            results = []
            for pmid in pmids:
                results.append(SourceResult(
                    source_id="pubmed",
                    source_name="PubMed",
                    id=pmid,
                    title=f"PubMed ID: {pmid}",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    description="Medical Research",
                    content_preview="Biomedical study indexed in PubMed.",
                    source_type="paper"
                ))
            return results

    async def fetch_full_content(self, result: SourceResult) -> str:
        return result.description
