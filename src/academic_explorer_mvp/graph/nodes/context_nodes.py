"""Context and graph boundary nodes."""

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.state import SearchState


QUERY_ENRICHMENT_DEFAULTS: dict[str, object] = {
    "stage": "idle",
    "original_query": None,
    "has_enough_context": None,
    "question": None,
    "reason": None,
    "answer": None,
    "proposed_query": None,
    "resolved_query": None,
    "confirmation_answer": None,
    "confirmation_status": None,
    "user_revision": None,
    "message": None,
    "round": 0,
}


PAUSE_STOP_REASONS = {
    "awaiting clarification answer",
    "awaiting query confirmation",
}


def initialize_context(state: SearchState) -> SearchState:
    """Initialize default state fields for the graph."""

    if "context" not in state:
        raise RuntimeError("SearchState must include a SearchContext.")

    new_state: SearchState = dict(state)
    new_state.setdefault("round_number", 0)
    new_state.setdefault("pending_queries", [])
    new_state.setdefault("query_history", [])
    new_state.setdefault("used_queries", [])
    new_state.setdefault("raw_results", [])
    new_state.setdefault("all_raw_results", [])
    new_state.setdefault("normalized_papers", [])
    new_state.setdefault("deduplicated_papers", [])
    new_state.setdefault("ranked_papers", [])
    new_state.setdefault("known_paper_ids", [])
    new_state.setdefault("last_new_paper_count", 0)
    new_state.setdefault("last_new_paper_ids", [])
    new_state.setdefault("last_new_useful_count", 0)
    new_state.setdefault("provider_errors", [])
    new_state.setdefault("stop_reason", None)
    new_state.setdefault("model_continue_reason", None)
    new_state["query_enrichment"] = _query_enrichment(new_state)
    return new_state


def finalize(state: SearchState) -> SearchState:
    """Return final state unchanged."""

    return state


def _context(state: SearchState) -> SearchContext:
    context = state.get("context")
    if context is None:
        raise RuntimeError("Search context is missing from graph state.")
    return context


def _query_enrichment(state: SearchState) -> dict[str, object]:
    """Return query enrichment substate with defaults applied."""

    enrichment = state.get("query_enrichment", {})
    if not isinstance(enrichment, dict):
        enrichment = {}
    return {**QUERY_ENRICHMENT_DEFAULTS, **enrichment}


def set_query_enrichment(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with query enrichment updates applied."""

    new_state: SearchState = dict(state)
    new_state["query_enrichment"] = {**_query_enrichment(state), **updates}
    return new_state


def _state_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _required_text(value: object, key: str) -> str:
    text = _state_text(value)
    if not text:
        raise RuntimeError(f"query_enrichment must include a non-empty `{key}` value.")
    return text
