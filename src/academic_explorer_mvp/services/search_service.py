"""Search service that calls academic providers."""

from dataclasses import dataclass, field

from academic_explorer_mvp.domain.paper import RawPaper
from academic_explorer_mvp.providers.base import SearchProvider


@dataclass(frozen=True)
class SearchBatch:
    """Raw results and provider errors from one graph round."""

    raw_papers: list[RawPaper]
    errors: list[str] = field(default_factory=list)


class SearchService:
    """Call each provider for each query."""

    def __init__(self, providers: list[SearchProvider]) -> None:
        self.providers = providers

    def search(self, queries: list[str], min_year: int, limit: int) -> SearchBatch:
        """Search all providers and return a simple batch."""

        raw_papers: list[RawPaper] = []
        errors: list[str] = []
        for query in queries:
            for provider in self.providers:
                try:
                    raw_papers.extend(provider.search(query=query, min_year=min_year, limit=limit))
                except Exception as exc:  # pragma: no cover - depends on external APIs.
                    errors.append(f"{provider.name} failed for query {query!r}: {exc}")
        return SearchBatch(raw_papers=raw_papers, errors=errors)
