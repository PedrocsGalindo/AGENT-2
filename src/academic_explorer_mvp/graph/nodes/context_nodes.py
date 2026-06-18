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


QUERY_PREVIEW_DEFAULTS: dict[str, object] = {
    "stage": "idle",
    "round": 0,
}


PAPER_FEEDBACK_DEFAULTS: dict[str, object] = {
    "stage": "idle",
    "answer": None,
    "pending_answer": None,
    "status": None,
    "message": None,
    "round": 0,
}


SEARCH_FEEDBACK_DEFAULTS: dict[str, object] = {
    "stage": "idle",
    "revised_topic": None,
    "positive_constraints": [],
    "negative_constraints": [],
    "query_strategy": None,
    "reason": None,
}


SEARCH_FILTERS_DEFAULTS: dict[str, object] = {
    "stage": "idle",
    "primary_intent": None,
    "conservative_filters": [],
    "expansive_filters": [],
    "negative_constraints": [],
    "not_inferred": [],
    "validation_priority": [],
    "reason": None,
}


PAUSE_STOP_REASONS = {
    "awaiting clarification answer",
    "awaiting query confirmation",
    "awaiting query preview",
    "awaiting paper feedback",
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
    new_state.setdefault("validated_papers", [])
    new_state.setdefault("relevant_papers", [])
    new_state.setdefault("excluded_papers", [])
    new_state.setdefault("validation_summary", None)
    new_state.setdefault("model_validation_summary", None)
    new_state.setdefault("judge_validation_summary", None)
    new_state.setdefault("judge_corrections_count", 0)
    new_state.setdefault("validation_counts", {})
    new_state.setdefault("known_paper_ids", [])
    new_state.setdefault("last_new_paper_count", 0)
    new_state.setdefault("last_new_paper_ids", [])
    new_state.setdefault("last_new_useful_count", 0)
    new_state.setdefault("provider_errors", [])
    new_state.setdefault("stop_reason", None)
    new_state.setdefault("model_continue_reason", None)
    new_state["query_enrichment"] = _query_enrichment(new_state)
    new_state["query_preview"] = _query_preview(new_state)
    new_state["paper_feedback"] = _paper_feedback(new_state)
    new_state["search_feedback"] = _search_feedback(new_state)
    new_state["search_filters"] = _search_filters(new_state)
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


def _paper_feedback(state: SearchState) -> dict[str, object]:
    """Return paper feedback substate with defaults applied."""

    feedback = state.get("paper_feedback", {})
    if not isinstance(feedback, dict):
        feedback = {}
    return {**PAPER_FEEDBACK_DEFAULTS, **feedback}


def _query_preview(state: SearchState) -> dict[str, object]:
    """Return query preview substate with defaults applied."""

    preview = state.get("query_preview", {})
    if not isinstance(preview, dict):
        preview = {}
    return {**QUERY_PREVIEW_DEFAULTS, **preview}


def _search_feedback(state: SearchState) -> dict[str, object]:
    """Return search feedback analysis substate with defaults applied."""

    feedback = state.get("search_feedback", {})
    if not isinstance(feedback, dict):
        feedback = {}
    return {**SEARCH_FEEDBACK_DEFAULTS, **feedback}


def _search_filters(state: SearchState) -> dict[str, object]:
    """Return semantic filter substate with defaults applied."""

    filters = state.get("search_filters", {})
    if not isinstance(filters, dict):
        filters = {}
    return {**SEARCH_FILTERS_DEFAULTS, **filters}


def set_query_enrichment(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with query enrichment updates applied."""

    new_state: SearchState = dict(state)
    new_state["query_enrichment"] = {**_query_enrichment(state), **updates}
    return new_state


def set_query_preview(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with query preview updates applied."""

    new_state: SearchState = dict(state)
    new_state["query_preview"] = {**_query_preview(state), **updates}
    return new_state


def set_paper_feedback(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with paper feedback updates applied."""

    new_state: SearchState = dict(state)
    new_state["paper_feedback"] = {**_paper_feedback(state), **updates}
    return new_state


def set_search_feedback(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with search feedback analysis updates applied."""

    new_state: SearchState = dict(state)
    new_state["search_feedback"] = {**_search_feedback(state), **updates}
    return new_state


def set_search_filters(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with semantic filter updates applied."""

    new_state: SearchState = dict(state)
    new_state["search_filters"] = {**_search_filters(state), **updates}
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
