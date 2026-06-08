"""Typed state passed through LangGraph."""

from typing import TypedDict

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import Paper, RankedPaper, RawPaper


class SearchState(TypedDict, total=False):
    """State for the explicit academic search graph."""

    context: SearchContext
    round_number: int
    pending_queries: list[str]
    query_history: list[list[str]]
    used_queries: list[str]
    raw_results: list[RawPaper]
    all_raw_results: list[RawPaper]
    normalized_papers: list[Paper]
    deduplicated_papers: list[Paper]
    ranked_papers: list[RankedPaper]
    known_paper_ids: list[str]
    last_new_paper_count: int
    last_new_paper_ids: list[str]
    last_new_useful_count: int
    provider_errors: list[str]
    stop_reason: str | None
    model_continue_reason: str | None
    query_enrichment: dict[str, object]
    paper_feedback: dict[str, object]
