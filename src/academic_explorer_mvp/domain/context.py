"""Search context created from CLI input."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchContext:
    """Compact user intent for one search session."""

    user_query: str
    min_year: int
    max_rounds: int
    limit: int
