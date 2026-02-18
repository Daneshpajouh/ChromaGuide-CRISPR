import requests
import re
import urllib.parse
from typing import List
from ..connectors.base import BaseSourceConnector, SourceResult

class LinkedInConnector(BaseSourceConnector):
    """
    Retrieves expert profiles from LinkedIn via targeted deep-web search.
    Uses 'site:linkedin.com/in/' filter to identify professionals.
    """

    async def search(self, query: str, limit: int = 5) -> List[SourceResult]:
        # Refine query for professionals
        refined_query = f'site:linkedin.com/in/ "{query}" (Professor OR "PhD" OR "Research Scientist" OR "Head of" OR "Lead")'
        # Fallback to DuckDuckGo HTML for robustness
        search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(refined_query)}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        }

        results = []
        try:
            # Sync request in async method (should be awaited in production, but ok for this scale)
            resp = requests.get(search_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                # Regex to extract results from DDG HTML
                links = re.findall(r'<a class="result__a" href="(https://[^"]+)">([^<]+)</a>', resp.text)
                snippets = re.findall(r'<a class="result__snippet" href="[^"]+">([^<]+)</a>', resp.text)

                for i, (url, title) in enumerate(links[:limit]):
                    if "linkedin.com/in/" in url:
                        snippet = snippets[i] if i < len(snippets) else "Professional Profile."
                        # Clean title
                        clean_title = title.replace(" | LinkedIn", "").strip()

                        results.append(SourceResult(
                            source_id="linkedin",
                            source_name="LinkedIn",
                            id=url,
                            title=clean_title,
                            url=url,
                            description=snippet,
                            content_preview=snippet,
                            source_type="professional",
                            relevance_score=0.9
                        ))
        except Exception as e:
            print(f"LinkedIn search error: {e}")

        return results

    async def fetch_full_content(self, result: SourceResult) -> str:
        """
        Fetches full content. For LinkedIn, we return the snippet and a directive to visit the profile.
        Authenticating purely via search scraper is not viable for full profile data.
        """
        return f"## LinkedIn Profile: {result.title}\n\n{result.description}\n\n[Professional Link]({result.url})"
