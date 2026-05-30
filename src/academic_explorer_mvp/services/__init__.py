"""Services used by the MVP graph."""

from academic_explorer_mvp.services.deduplicator import PaperDeduplicator
from academic_explorer_mvp.services.normalizer import PaperNormalizer
from academic_explorer_mvp.services.query_planner import QueryPlanner
from academic_explorer_mvp.services.ranker import PaperRanker
from academic_explorer_mvp.services.search_service import SearchService

__all__ = [
    "PaperDeduplicator",
    "PaperNormalizer",
    "PaperRanker",
    "QueryPlanner",
    "SearchService",
]
