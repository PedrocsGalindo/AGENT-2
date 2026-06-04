"""Search planning and provider search nodes."""

from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph.nodes.context_nodes import _context
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
            ranked_papers=state.get("ranked_papers", []),
            used_queries=state.get("used_queries", []),
        )

    query_history = list(state.get("query_history", []))
    query_history.append(queries)

    new_state: SearchState = dict(state)
    new_state["pending_queries"] = queries
    new_state["query_history"] = query_history
    new_state["used_queries"] = [*state.get("used_queries", []), *queries]
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


def rank_papers(state: SearchState, ranker: PaperRanker) -> SearchState:
    """Rank papers deterministically after deduplication."""

    ranked = ranker.rank(state.get("deduplicated_papers", []), _context(state))
    new_ids = set(state.get("last_new_paper_ids", []))

    new_useful = [
        item
        for item in ranked
        if item.paper.id in new_ids and item.score >= ranker.good_score_threshold
    ]

    new_state: SearchState = dict(state)
    new_state["ranked_papers"] = ranked
    new_state["last_new_useful_count"] = len(new_useful)
    return new_state
