import requests
from typing import List
from ..connectors.base import BaseSourceConnector, SourceResult

class HFPapersConnector(BaseSourceConnector):
    """
    Fetches the absolute latest AI papers from Hugging Face's Daily Papers endpoint.
    Ensures 'Zero-Latency Intelligence' on current research.
    """

    API_ENDPOINT = "https://huggingface.co/api/daily_papers"

    async def search(self, query: str, limit: int = 5) -> List[SourceResult]:
        # Note: 'query' argument is kept for interface consistency,
        # but this connector prioritizes *latest* global papers, filtered locally if needed.

        results = []
        try:
            resp = requests.get(self.API_ENDPOINT, timeout=10)
            if resp.status_code == 200:
                papers = resp.json()

                # Filter locally if query is specific, otherwise take top trending
                query_terms = query.replace('"', '').lower().split()

                count = 0
                for p in papers:
                    title = p.get('title', 'Unknown Paper')
                    summary = p.get('summary', '') or "No summary available."
                    paper_id = p.get('paper', {}).get('id', '')

                    # Basic relevancy check
                    text_blob = (title + " " + summary).lower()
                    if any(term in text_blob for term in query_terms) or not query_terms:
                        url = f"https://huggingface.co/papers/{paper_id}" if paper_id else "https://huggingface.co/papers"

                        results.append(SourceResult(
                            source_id="huggingface_daily",
                            source_name="HuggingFace Daily",
                            id=paper_id,
                            title=f"[DAILY PAPER] {title}",
                            url=url,
                            description=summary[:500],
                            content_preview=summary[:500],
                            source_type="paper",
                            relevance_score=0.95
                        ))
                        count += 1
                        if count >= limit:
                            break

        except Exception as e:
            print(f"HF Daily Papers error: {e}")

        return results

    async def fetch_full_content(self, result: SourceResult) -> str:
        """
        Fetches full content. We return the extended summary and link.
        """
        return f"## {result.title}\n\n**Abstract/Summary**:\n{result.description}\n\n[View on Hugging Face]({result.url})"
