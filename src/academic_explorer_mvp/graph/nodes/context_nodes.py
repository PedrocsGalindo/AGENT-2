"""Context and graph boundary nodes."""

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.utils.text import clean_text


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


STATE_DEFAULTS: dict[str, object] = {
    "round_number": 0,
    "pending_queries": [],
    "query_history": [],
    "used_queries": [],
    "raw_results": [],
    "all_raw_results": [],
    "normalized_papers": [],
    "deduplicated_papers": [],
    "ranked_papers": [],
    "validated_papers": [],
    "relevant_papers": [],
    "excluded_papers": [],
    "validation_summary": None,
    "model_validation_summary": None,
    "judge_validation_summary": None,
    "judge_corrections_count": 0,
    "validation_counts": {},
    "known_paper_ids": [],
    "last_new_paper_count": 0,
    "last_new_paper_ids": [],
    "last_new_useful_count": 0,
    "provider_errors": [],
    "stop_reason": None,
    "model_continue_reason": None,
}


def initialize_context(state: SearchState) -> SearchState:
    """Initialize default state fields for the graph."""

    if "context" not in state:
        raise RuntimeError("SearchState must include a SearchContext.")

    new_state: SearchState = dict(state)
    for key, value in STATE_DEFAULTS.items():
        new_state.setdefault(key, value.copy() if isinstance(value, list | dict) else value)
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


def _substate(
    state: SearchState,
    key: str,
    defaults: dict[str, object],
) -> dict[str, object]:
    value = state.get(key, {})
    if not isinstance(value, dict):
        value = {}
    return {**defaults, **value}


def _set_substate(
    state: SearchState,
    key: str,
    defaults: dict[str, object],
    **updates: object,
) -> SearchState:
    new_state: SearchState = dict(state)
    new_state[key] = {**_substate(state, key, defaults), **updates}
    return new_state


def _query_enrichment(state: SearchState) -> dict[str, object]:
    """Return query enrichment substate with defaults applied."""

    return _substate(state, "query_enrichment", QUERY_ENRICHMENT_DEFAULTS)


def _paper_feedback(state: SearchState) -> dict[str, object]:
    """Return paper feedback substate with defaults applied."""

    return _substate(state, "paper_feedback", PAPER_FEEDBACK_DEFAULTS)


def _query_preview(state: SearchState) -> dict[str, object]:
    """Return query preview substate with defaults applied."""

    return _substate(state, "query_preview", QUERY_PREVIEW_DEFAULTS)


def _search_feedback(state: SearchState) -> dict[str, object]:
    """Return search feedback analysis substate with defaults applied."""

    return _substate(state, "search_feedback", SEARCH_FEEDBACK_DEFAULTS)


def _search_filters(state: SearchState) -> dict[str, object]:
    """Return semantic filter substate with defaults applied."""

    return _substate(state, "search_filters", SEARCH_FILTERS_DEFAULTS)


def set_query_enrichment(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with query enrichment updates applied."""

    return _set_substate(state, "query_enrichment", QUERY_ENRICHMENT_DEFAULTS, **updates)


def set_query_preview(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with query preview updates applied."""

    return _set_substate(state, "query_preview", QUERY_PREVIEW_DEFAULTS, **updates)


def set_paper_feedback(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with paper feedback updates applied."""

    return _set_substate(state, "paper_feedback", PAPER_FEEDBACK_DEFAULTS, **updates)


def set_search_feedback(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with search feedback analysis updates applied."""

    return _set_substate(state, "search_feedback", SEARCH_FEEDBACK_DEFAULTS, **updates)


def set_search_filters(state: SearchState, **updates: object) -> SearchState:
    """Return a new state with semantic filter updates applied."""

    return _set_substate(state, "search_filters", SEARCH_FILTERS_DEFAULTS, **updates)


def _state_text(value: object) -> str:
    return clean_text(value)


def _required_text(value: object, key: str) -> str:
    text = _state_text(value)
    if not text:
        raise RuntimeError(f"query_enrichment must include a non-empty `{key}` value.")
    return text
