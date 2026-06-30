"""Deterministic and explainable paper ranking."""



from datetime import datetime
import math

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import Paper, RankedPaper
from academic_explorer_mvp.utils.text import TOKEN_RE, find_constraint_hits


class PaperRanker:
    """Rank papers with transparent handcrafted signals."""

    good_score_threshold = 6.0

    def rank(
        self,
        papers: list[Paper],
        context: SearchContext,
        negative_constraints: list[str] | None = None,
    ) -> list[RankedPaper]:
        """Rank papers for the current query context."""

        max_citations = max((paper.citation_count or 0 for paper in papers), default=0)
        max_citation_log = math.log1p(max_citations) or 1.0
        query_terms = self._terms(context.user_query)
        negative_constraints = negative_constraints or []

        ranked = [
            self._rank_one(
                paper=paper,
                context=context,
                query_terms=query_terms,
                max_citation_log=max_citation_log,
                negative_constraints=negative_constraints,
            )
            for paper in papers
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)

    def _rank_one(
        self,
        paper: Paper,
        context: SearchContext,
        query_terms: list[str],
        max_citation_log: float,
        negative_constraints: list[str],
    ) -> RankedPaper:
        title = (paper.title or "").lower()
        abstract = (paper.abstract or "").lower()
        combined_text = f"{title}\n{abstract}"

        title_hits = sum(1 for term in query_terms if term in title)
        abstract_hits = sum(1 for term in query_terms if term in abstract)
        title_score = 4.0 * (title_hits / max(len(query_terms), 1))
        abstract_score = 2.0 * (abstract_hits / max(len(query_terms), 1))
        year_score = self._year_score(paper.year, context.min_year)
        citation_score = 2.0 * (math.log1p(paper.citation_count or 0) / max_citation_log)
        metadata_score = 0.0

        reasons: list[str] = []
        if title_hits:
            reasons.append(f"{title_hits} query terms in title")
        if abstract_hits:
            reasons.append(f"{abstract_hits} query terms in abstract")
        if year_score:
            reasons.append("publication year matches recency filter")
        if citation_score >= 1.0:
            reasons.append("relative citation count is useful")
        if paper.abstract:
            metadata_score += 0.5
            reasons.append("has abstract")
        if paper.url:
            metadata_score += 0.25
            reasons.append("has URL")
        if paper.doi:
            metadata_score += 0.25
            reasons.append("has DOI")

        score = title_score + abstract_score + year_score + citation_score + metadata_score
        negative_hits = find_constraint_hits(combined_text, negative_constraints)
        if negative_hits:
            penalty = 4.0 * len(negative_hits)
            score = max(0.0, score - penalty)
            reasons.append(
                "penalized by negative constraints: " + ", ".join(negative_hits[:4])
            )
        if not reasons:
            reasons.append("weak metadata match")

        return RankedPaper(paper=paper, score=round(score, 3), reasons=reasons)

    def _year_score(self, year: int | None, min_year: int) -> float:
        if year is None or year < min_year:
            return 0.0
        current_year = datetime.now().year
        span = max(current_year - min_year, 1)
        return 1.0 + min((year - min_year) / span, 1.0)

    def _terms(self, query: str) -> list[str]:
        return [term for term in TOKEN_RE.findall(query.lower()) if len(term) > 2]

