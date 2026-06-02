"""Paper entities used by the MVP flow."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RawPaper:
    """Raw provider result before normalization."""

    source: str
    source_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class Paper:
    """Normalized paper used by services and graph nodes."""

    id: str
    title: str
    abstract: str | None
    year: int | None
    authors: list[str]
    source: str
    source_id: str
    url: str | None
    doi: str | None
    citation_count: int

    def text(self) -> str:
        """Return searchable text used by the simple deterministic ranker."""

        return f"{self.title or ''}\n{self.abstract or ''}".strip()


@dataclass(frozen=True)
class RankedPaper:
    """Paper with an explainable score."""

    paper: Paper
    score: float
    reasons: list[str] = field(default_factory=list)
