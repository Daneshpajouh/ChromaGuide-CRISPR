import httpx
from typing import List, Optional
from .base import BaseSourceConnector, SourceResult

class HuggingFaceConnector(BaseSourceConnector):
    """Search Hugging Face Hub for models and datasets."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.base_url = "https://huggingface.co/api"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def search(self, query: str, limit: int = 20) -> List[SourceResult]:
        results = []
        async with httpx.AsyncClient(headers=self.headers, timeout=10) as client:
            url = f"{self.base_url}/models"
            params = {"search": query, "sort": "downloads", "direction": -1, "limit": limit}

            response = await client.get(url, params=params)
            if response.status_code != 200: return []

            data = response.json()
            for model in data:
                results.append(SourceResult(
                    source_id="huggingface",
                    source_name="Hugging Face",
                    id=model["id"],
                    title=model["id"],
                    url=f"https://huggingface.co/{model['id']}",
                    description=model.get("pipeline_tag", "model"),
                    content_preview=f"Downloads: {model.get('downloads', 0)}, Likes: {model.get('likes', 0)}",
                    metadata={
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                        "task": model.get("pipeline_tag")
                    },
                    source_type="model"
                ))
        return results

    async def fetch_full_content(self, result: SourceResult) -> str:
        model_id = result.id
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            return response.text if response.status_code == 200 else result.description
