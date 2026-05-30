"""Semantic Scholar provider implemented directly for the MVP."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from academic_explorer_mvp.domain.paper import RawPaper


class SemanticScholarProvider:
    """Search papers in Semantic Scholar."""

    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 20) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, min_year: int, limit: int) -> list[RawPaper]:
        """Search Semantic Scholar and keep provider payloads intact."""

        params = {
            "query": query,
            "limit": max(1, min(limit, 100)),
            "fields": "paperId,title,abstract,year,citationCount,url,externalIds,authors",
            "year": f"{min_year}-",
        }
        headers = {"User-Agent": "academic-explorer-mvp/0.1"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        url = f"{self.base_url}?{urlencode(params)}"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Semantic Scholar request failed: {exc}") from exc

        return [
            RawPaper(source=self.name, source_id=str(item.get("paperId") or ""), payload=item)
            for item in payload.get("data", [])
        ]
