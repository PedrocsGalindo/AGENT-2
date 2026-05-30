"""OpenAlex provider implemented directly for the MVP."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from academic_explorer_mvp.domain.paper import RawPaper


class OpenAlexProvider:
    """Search papers in OpenAlex."""

    name = "openalex"
    base_url = "https://api.openalex.org/works"

    def __init__(self, mailto: str | None = None, timeout_seconds: int = 20) -> None:
        self.mailto = mailto
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, min_year: int, limit: int) -> list[RawPaper]:
        """Search OpenAlex and keep provider payloads intact."""

        params = {
            "search": query,
            "per-page": max(1, min(limit, 50)),
            "filter": f"from_publication_date:{min_year}-01-01",
        }
        if self.mailto:
            params["mailto"] = self.mailto

        url = f"{self.base_url}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "academic-explorer-mvp/0.1"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAlex request failed: {exc}") from exc

        return [
            RawPaper(source=self.name, source_id=str(item.get("id") or ""), payload=item)
            for item in payload.get("results", [])
        ]
