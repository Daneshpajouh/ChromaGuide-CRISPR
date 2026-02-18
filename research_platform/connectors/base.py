from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SourceResult:
    """Unified result format across all sources."""
    source_id: str  # "github", "huggingface", "arxiv", etc.
    source_name: str
    id: str  # Unique ID in that source
    title: str
    url: str
    description: str
    content_preview: str  # First 500 chars
    full_content: Optional[str] = None  # Fetched on demand
    metadata: Dict = None  # Source-specific (stars, authors, date, etc.)
    relevance_score: float = 0.0  # 0-1, computed by ranking engine
    fetched_at: str = None
    source_type: str = ""  # "code", "paper", "model", "dataset", "documentation"

class BaseSourceConnector(ABC):
    """All connectors inherit from this."""

    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 100):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.request_count = 0

    @abstractmethod
    async def search(self, query: str, limit: int = 20) -> List[SourceResult]:
        """Search the source."""
        pass

    @abstractmethod
    async def fetch_full_content(self, result: SourceResult) -> str:
        """Fetch full content (code, paper, etc.)."""
        pass

    def _check_rate_limit(self):
        """Enforce rate limiting."""
        # Simple placeholder for rate limiting logic
        self.request_count += 1
