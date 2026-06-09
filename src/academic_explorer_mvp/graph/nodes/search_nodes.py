"""Search planning and provider search nodes."""

import re

from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph.nodes.context_nodes import _context, set_query_preview
from academic_explorer_mvp.services.query_planner import QueryPlanner
from academic_explorer_mvp.services.search_service import SearchService

from academic_explorer_mvp.graph.nodes.context_nodes import _context
from academic_explorer_mvp.services.deduplicator import PaperDeduplicator
from academic_explorer_mvp.services.normalizer import PaperNormalizer
from academic_explorer_mvp.services.ranker import PaperRanker

def plan_queries(state: SearchState, planner: QueryPlanner) -> SearchState:
    """Plan initial or refined queries with the local model."""

    context = _context(state)
    if state.get("round_number", 0) == 0:
        queries = planner.plan_initial_queries(context)
    else:
        queries = planner.refine_queries(
            context=context,
            validated_papers=state.get("validated_papers", []),
            used_queries=state.get("used_queries", []),
            search_feedback=state.get("search_feedback", {}) or {},
        )

    query_history = list(state.get("query_history", []))
    query_history.append(queries)

    new_state: SearchState = dict(state)
    new_state["pending_queries"] = queries
    new_state["query_history"] = query_history
    new_state["used_queries"] = [*state.get("used_queries", []), *queries]
    new_state = set_query_preview(
        new_state,
        stage="awaiting_query_preview",
        round=state.get("round_number", 0) + 1,
    )
    new_state["stop_reason"] = "awaiting query preview"
    return new_state


def search_papers(state: SearchState, search_service: SearchService) -> SearchState:
    """Search papers in OpenAlex and Semantic Scholar."""

    context = _context(state)
    queries = state.get("pending_queries", [])
    batch = search_service.search(queries=queries, min_year=context.min_year, limit=context.limit)

    new_state: SearchState = dict(state)
    new_state["round_number"] = state.get("round_number", 0) + 1
    new_state["raw_results"] = batch.raw_papers
    new_state["all_raw_results"] = [*state.get("all_raw_results", []), *batch.raw_papers]
    new_state["provider_errors"] = [*state.get("provider_errors", []), *batch.errors]
    new_state = set_query_preview(new_state, stage="idle")
    return new_state

def normalize_papers(state: SearchState, normalizer: PaperNormalizer) -> SearchState:
    """Normalize all cumulative raw papers."""

    normalized = normalizer.normalize_many(state.get("all_raw_results", []))
    new_state: SearchState = dict(state)
    new_state["normalized_papers"] = normalized
    return new_state


def deduplicate_papers(state: SearchState, deduplicator: PaperDeduplicator) -> SearchState:
    """Deduplicate normalized papers and count newly discovered papers."""

    previous_ids = set(state.get("known_paper_ids", []))
    deduplicated = deduplicator.deduplicate(state.get("normalized_papers", []))
    current_ids = {paper.id for paper in deduplicated}
    new_ids = current_ids - previous_ids

    new_state: SearchState = dict(state)
    new_state["deduplicated_papers"] = deduplicated
    new_state["known_paper_ids"] = sorted(current_ids)
    new_state["last_new_paper_count"] = len(new_ids)
    new_state["last_new_paper_ids"] = sorted(new_ids)
    return new_state


def validate_papers(state: SearchState, planner: QueryPlanner) -> SearchState:
    """Validate papers semantically against the current search intent."""

    context = _context(state)
    papers = state.get("deduplicated_papers", [])
    result = planner.validate_papers(
        context=context,
        papers=papers,
        search_feedback=state.get("search_feedback", {}) or {},
    )
    validations_by_id = {item.paper_id: item for item in result.validated_papers}
    search_feedback = state.get("search_feedback", {}) or {}
    negative_constraints = search_feedback.get("negative_constraints", [])
    if not isinstance(negative_constraints, list):
        negative_constraints = []

    validated: list[dict[str, object]] = []
    for paper in papers:
        validation = validations_by_id.get(paper.id)
        if validation is None:
            item = {
                "paper": paper,
                "paper_id": paper.id,
                "relevance": "reject",
                "decision": "exclude",
                "relevance_reason": "",
                "mismatch_reason": "The model did not return a validation for this paper.",
                "useful_for": "",
            }
        else:
            item = {
                "paper": paper,
                "paper_id": paper.id,
                "relevance": validation.relevance,
                "decision": validation.decision,
                "relevance_reason": validation.relevance_reason,
                "mismatch_reason": validation.mismatch_reason,
                "useful_for": validation.useful_for,
            }
        validated.append(_apply_negative_constraints(item, negative_constraints))

    validated = sorted(validated, key=_validation_sort_key)
    relevant = [item for item in validated if item.get("decision") == "include"]
    excluded = [item for item in validated if item.get("decision") == "exclude"]
    new_ids = set(state.get("last_new_paper_ids", []))
    useful_relevances = {"high", "medium"}
    new_useful = [
        item
        for item in relevant
        if getattr(item.get("paper"), "id", None) in new_ids
        and item.get("relevance") in useful_relevances
    ]

    new_state: SearchState = dict(state)
    new_state["validated_papers"] = validated
    new_state["relevant_papers"] = relevant
    new_state["excluded_papers"] = excluded
    new_state["validation_summary"] = result.summary
    new_state["ranked_papers"] = []
    new_state["last_new_useful_count"] = len(new_useful)
    return new_state

def _validation_sort_key(item: dict[str, object]) -> tuple[int, int]:
    relevance_order = {"high": 0, "medium": 1, "low": 2, "reject": 3}
    paper = item.get("paper")
    year = getattr(paper, "year", None) or 0
    return (relevance_order.get(str(item.get("relevance")), 3), -year)


def _apply_negative_constraints(
    item: dict[str, object],
    negative_constraints: list[object],
) -> dict[str, object]:
    hits = _negative_hits(item.get("paper"), negative_constraints)
    if not hits:
        return item

    new_item = dict(item)
    new_item["relevance"] = "reject"
    new_item["decision"] = "exclude"
    reason = "Matches excluded terms: " + ", ".join(hits[:4])
    existing = str(new_item.get("mismatch_reason") or "").strip()
    new_item["mismatch_reason"] = f"{existing} {reason}".strip()
    return new_item


def _negative_hits(paper: object, constraints: list[object]) -> list[str]:
    title = str(getattr(paper, "title", "") or "").lower()
    abstract = str(getattr(paper, "abstract", "") or "").lower()
    text = f"{title}\n{abstract}"
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text))
    hits: list[str] = []
    seen: set[str] = set()

    for constraint in constraints:
        label = " ".join(str(constraint).split())
        normalized_constraint = " ".join(re.findall(r"[a-z0-9]+", label.lower()))
        if not normalized_constraint or normalized_constraint in seen:
            continue
        if " " in normalized_constraint:
            matched = normalized_constraint in normalized_text
        else:
            matched = re.search(rf"\b{re.escape(normalized_constraint)}\b", normalized_text)
        if matched:
            seen.add(normalized_constraint)
            hits.append(label)
    return hits
