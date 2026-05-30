"""Simple deterministic deduplication."""

from __future__ import annotations

import re

from academic_explorer_mvp.domain.paper import Paper


NON_WORD_RE = re.compile(r"[^a-z0-9]+")


class PaperDeduplicator:
    """Deduplicate by DOI, provider id, then title and year."""

    def deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """Return papers with simple duplicate keys removed."""

        seen: set[str] = set()
        output: list[Paper] = []
        for paper in papers:
            key = self._key(paper)
            if key in seen:
                continue
            seen.add(key)
            output.append(paper)
        return output

    def _key(self, paper: Paper) -> str:
        if paper.doi:
            return f"doi:{paper.doi.lower().strip()}"
        if paper.source and paper.source_id:
            return f"source:{paper.source}:{paper.source_id}"

        normalized_title = NON_WORD_RE.sub(" ", paper.title.lower()).strip()
        return f"title:{normalized_title}:{paper.year or 'unknown'}"
