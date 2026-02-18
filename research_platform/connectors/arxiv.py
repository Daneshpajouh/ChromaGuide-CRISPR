import httpx
from typing import List
import xml.etree.ElementTree as ET
from .base import BaseSourceConnector, SourceResult

class ArxivConnector(BaseSourceConnector):
    """Search arXiv for preprints."""

    def __init__(self):
        super().__init__()
        self.base_url = "http://export.arxiv.org/api/query"

    async def search(self, query: str, limit: int = 20) -> List[SourceResult]:
        arxiv_query = f'all:"{query}"'
        async with httpx.AsyncClient(timeout=15) as client:
            params = {
                "search_query": arxiv_query,
                "start": 0,
                "max_results": limit,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            response = await client.get(self.base_url, params=params)
            if response.status_code != 200: return []
            return self._parse_arxiv_response(response.text)

    def _parse_arxiv_response(self, xml_text: str) -> List[SourceResult]:
        results = []
        try:
            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[1]
                title = entry.find("atom:title", ns).text.replace("\n", " ").strip()
                summary = entry.find("atom:summary", ns).text.replace("\n", " ").strip()

                results.append(SourceResult(
                    source_id="arxiv",
                    source_name="arXiv",
                    id=arxiv_id,
                    title=title,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    description="Paper",
                    content_preview=summary[:300],
                    source_type="paper"
                ))
        except: pass
        return results

    async def fetch_full_content(self, result: SourceResult) -> str:
        return result.content_preview
