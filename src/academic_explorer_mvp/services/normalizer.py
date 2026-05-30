"""Normalize provider payloads into the MVP Paper shape."""

from __future__ import annotations

import hashlib
from typing import Any

from academic_explorer_mvp.domain.paper import Paper, RawPaper


class PaperNormalizer:
    """Provider-specific normalization kept in one small class."""

    def normalize_many(self, raw_papers: list[RawPaper]) -> list[Paper]:
        """Normalize many raw papers, skipping payloads without a title."""

        papers: list[Paper] = []
        for raw_paper in raw_papers:
            paper = self.normalize(raw_paper)
            if paper is not None:
                papers.append(paper)
        return papers

    def normalize(self, raw_paper: RawPaper) -> Paper | None:
        """Normalize one provider payload."""

        if raw_paper.source == "openalex":
            return self._normalize_openalex(raw_paper)
        if raw_paper.source == "semantic_scholar":
            return self._normalize_semantic_scholar(raw_paper)
        return None

    def _normalize_openalex(self, raw_paper: RawPaper) -> Paper | None:
        data = raw_paper.payload
        title = data.get("title") or data.get("display_name")
        if not title:
            return None

        ids = data.get("ids") or {}
        doi = data.get("doi") or ids.get("doi")
        url = self._openalex_url(data) or ids.get("openalex")
        authors = [
            ((item.get("author") or {}).get("display_name") or "").strip()
            for item in data.get("authorships", [])
        ]

        return Paper(
            id=self._stable_id(doi or raw_paper.source_id or title),
            title=title,
            abstract=self._abstract_from_openalex(data.get("abstract_inverted_index")),
            year=data.get("publication_year"),
            authors=[author for author in authors if author],
            source=raw_paper.source,
            source_id=raw_paper.source_id,
            url=url,
            doi=doi,
            citation_count=int(data.get("cited_by_count") or 0),
        )

    def _normalize_semantic_scholar(self, raw_paper: RawPaper) -> Paper | None:
        data = raw_paper.payload
        title = data.get("title")
        if not title:
            return None

        external_ids = data.get("externalIds") or {}
        doi = external_ids.get("DOI")
        authors = [
            (author.get("name") or "").strip()
            for author in data.get("authors", [])
            if isinstance(author, dict)
        ]

        return Paper(
            id=self._stable_id(doi or raw_paper.source_id or title),
            title=title,
            abstract=data.get("abstract"),
            year=data.get("year"),
            authors=[author for author in authors if author],
            source=raw_paper.source,
            source_id=raw_paper.source_id,
            url=data.get("url"),
            doi=doi,
            citation_count=int(data.get("citationCount") or 0),
        )

    def _stable_id(self, value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()

    def _openalex_url(self, data: dict[str, Any]) -> str | None:
        location = data.get("primary_location") or {}
        return location.get("landing_page_url")

    def _abstract_from_openalex(self, index: dict[str, list[int]] | None) -> str | None:
        if not index:
            return None

        positions: dict[int, str] = {}
        for word, word_positions in index.items():
            for position in word_positions:
                positions[position] = word
        return " ".join(positions[position] for position in sorted(positions))
