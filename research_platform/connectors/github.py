import httpx
from typing import List
from .base import BaseSourceConnector, SourceResult

class GitHubConnector(BaseSourceConnector):
    """Search GitHub repositories and files."""

    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.github.com"
        self.headers = {"Authorization": f"token {api_key}"} if api_key else {}

    async def search(self, query: str, limit: int = 20) -> List[SourceResult]:
        results = []
        async with httpx.AsyncClient(headers=self.headers, timeout=10) as client:
            # Search repositories
            url = f"{self.base_url}/search/repositories"
            params = {"q": query, "sort": "stars", "order": "desc", "per_page": limit}

            response = await client.get(url, params=params)
            if response.status_code != 200: return []

            data = response.json()
            for repo in data.get("items", []):
                results.append(SourceResult(
                    source_id="github",
                    source_name="GitHub",
                    id=str(repo["id"]),
                    title=repo["full_name"],
                    url=repo["html_url"],
                    description=repo["description"] or "",
                    content_preview=f"Stars: {repo['stargazers_count']}, Lang: {repo['language']}",
                    metadata={
                        "stars": repo["stargazers_count"],
                        "language": repo["language"],
                        "updated_at": repo["updated_at"]
                    },
                    source_type="code"
                ))
        return results

    async def fetch_full_content(self, result: SourceResult) -> str:
        # Implementation to fetch README
        repo = result.title
        url = f"{self.base_url}/repos/{repo}/readme"
        async with httpx.AsyncClient(headers=self.headers, timeout=10) as client:
            response = await client.get(url, headers={"Accept": "application/vnd.github.v3.raw"})
            return response.text if response.status_code == 200 else result.description
