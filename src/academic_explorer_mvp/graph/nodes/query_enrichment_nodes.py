"""Nodes for the initial query clarification flow."""

from dataclasses import replace

from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph.nodes.context_nodes import (
    PAUSE_STOP_REASONS,
    _context,
    _query_enrichment,
    _required_text,
    set_query_enrichment,
    _state_text,
)
from academic_explorer_mvp.services.query_planner import QueryPlanner


def route_after_context_initialization(state: SearchState) -> str:
    """Route a fresh or resumed state to the next query enrichment step."""

    enrichment = _query_enrichment(state)
    stage = enrichment.get("stage")

    if stage in {"awaiting_query_confirmation", "unclear_confirmation"}:
        if _state_text(enrichment.get("confirmation_answer")):
            return "handle_query_confirmation_or_revision"
        return "finalize"

    if stage == "awaiting_clarification_answer":
        if _state_text(enrichment.get("answer")):
            return "rewrite_user_query_after_clarification"
        return "finalize"

    if _state_text(enrichment.get("confirmation_answer")):
        return "handle_query_confirmation_or_revision"

    if _state_text(enrichment.get("answer")) and enrichment.get("question"):
        return "rewrite_user_query_after_clarification"

    return "assess_initial_query"


def assess_initial_query(state: SearchState, planner: QueryPlanner) -> SearchState:
    """Assess whether the initial user topic is ready for academic search."""

    context = _context(state)
    enrichment = _query_enrichment(state)
    assessment = planner.assess_initial_query(context)
    original_query = _state_text(enrichment.get("original_query")) or context.user_query

    if assessment.can_search:
        new_state = set_query_enrichment(
            state,
            stage="ready_to_search",
            original_query=original_query,
            can_search=True,
            reason=assessment.reason,
        )
        if new_state.get("stop_reason") in PAUSE_STOP_REASONS:
            new_state["stop_reason"] = None
        return new_state

    new_state = set_query_enrichment(
        state,
        stage="awaiting_clarification_answer",
        original_query=original_query,
        can_search=False,
        question=assessment.question,
        reason=assessment.reason,
        answer=None,
        proposed_query=None,
        confirmation_answer=None,
        confirmation_status=None,
        user_revision=None,
        message=None,
        round=int(enrichment.get("round") or 0) + 1,
    )
    new_state["stop_reason"] = "awaiting clarification answer"
    return new_state


def route_after_initial_assessment(state: SearchState) -> str:
    stage = _query_enrichment(state).get("stage")

    if stage == "ready_to_search":
        return "ready_to_search"
    return "needs_clarification"


def rewrite_user_query_after_clarification(
    state: SearchState,
    planner: QueryPlanner,
) -> SearchState:
    """Create a proposed user topic after receiving a clarification answer."""

    context = _context(state)
    enrichment = _query_enrichment(state)

    initial_query = _state_text(enrichment.get("original_query")) or context.user_query
    question = _required_text(enrichment.get("question"), "question")
    question_reason = _state_text(enrichment.get("reason"))
    user_answer = _required_text(enrichment.get("answer"), "answer")

    rewrite = planner.rewrite_user_query(
        initial_query=initial_query,
        question=question,
        question_reason=question_reason,
        user_answer=user_answer,
    )

    proposed_query = rewrite.proposed_query

    message = (
        "Com base na sua resposta, a busca ficaria:\n\n"
        f'"{proposed_query}"\n\n'
        'Se estiver boa, responda "sim".\n'
        "Se quiser melhorar, escreva uma versão mais específica.\n"
        "Tente incluir, se fizer sentido: área, modalidade, "
        "objetivo, assuntos relacionados ou restrições."
    )

    new_state = set_query_enrichment(
        state,
        stage="awaiting_query_confirmation",
        proposed_query=proposed_query,
        message=message,
        confirmation_answer=None,
        confirmation_status=None,
        user_revision=None,
    )

    new_state["stop_reason"] = "awaiting query confirmation"
    return new_state


def handle_query_confirmation_or_revision(
    state: SearchState,
    planner: QueryPlanner,
) -> SearchState:
    """Handle acceptance, unclear confirmation, or a user-written revision."""

    enrichment = _query_enrichment(state)
    answer = _state_text(enrichment.get("confirmation_answer"))
    status = interpret_query_confirmation(answer)

    if status == "accepted":
        return set_query_enrichment(
            state,
            confirmation_answer=None,
            confirmation_status="accepted",
            user_revision=None,
        )

    if status == "unclear":
        new_state = set_query_enrichment(
            state,
            stage="unclear_confirmation",
            confirmation_answer=None,
            confirmation_status="unclear",
            message=(
                'Responda "sim" para aceitar ou escreva uma versao mais especifica '
                "da busca."
            ),
        )
        new_state["stop_reason"] = "awaiting query confirmation"
        return new_state

    return _rewrite_from_user_revision(state, planner, answer)


def route_after_query_confirmation(state: SearchState) -> str:
    status = _query_enrichment(state).get("confirmation_status")

    if status == "accepted":
        return "commit_query"
    return "wait_for_user"


def commit_enriched_query(state: SearchState) -> SearchState:
    """Commit the accepted proposed topic into SearchContext.user_query."""

    context = _context(state)
    enrichment = _query_enrichment(state)
    proposed_query = _required_text(enrichment.get("proposed_query"), "proposed_query")

    new_state: SearchState = dict(state)
    new_state["context"] = replace(context, user_query=proposed_query)
    new_state = set_query_enrichment(
        new_state,
        stage="committed",
        resolved_query=proposed_query,
        confirmation_answer=None,
        confirmation_status=None,
        user_revision=None,
        message=None,
    )
    if new_state.get("stop_reason") in PAUSE_STOP_REASONS:
        new_state["stop_reason"] = None
    return new_state


def interpret_query_confirmation(user_message: str) -> str:
    text = user_message.strip().lower()

    positive_answers = {
        "sim",
        "s",
        "ok",
        "pode seguir",
        "segue",
        "correto",
        "isso",
        "isso mesmo",
        "ta bom",
        "t\u00e1 bom",
    }

    if text in positive_answers:
        return "accepted"

    if len(text) >= 8:
        return "revised"

    return "unclear"

def _rewrite_from_user_revision(
    state: SearchState,
    planner: QueryPlanner,
    user_revision: str,
) -> SearchState:
    context = _context(state)
    enrichment = _query_enrichment(state)

    rewrite = planner.rewrite_from_user_revision(
        initial_query=_state_text(enrichment.get("original_query")) or context.user_query,
        proposed_query=_required_text(enrichment.get("proposed_query"), "proposed_query"),
        clarification_question=_required_text(enrichment.get("question"), "question"),
        clarification_reason=_state_text(enrichment.get("reason")),
        clarification_answer=_required_text(enrichment.get("answer"), "answer"),
        user_revision=user_revision,
    )

    message = (
        "Com base na sua resposta, a busca ficaria:\n\n"
        f'"{rewrite.proposed_query}"\n\n'
        'Se estiver boa, responda "sim".\n'
        "Se quiser melhorar, escreva uma versão mais específica.\n"
        "Tente incluir, se fizer sentido: área, modalidade, "
        "objetivo, assuntos relacionados ou restrições."
    )

    new_state = set_query_enrichment(
        state,
        stage="awaiting_query_confirmation",
        proposed_query=rewrite.proposed_query,
        confirmation_answer=None,
        confirmation_status="revised",
        user_revision=None,
        message=message,
        can_search=rewrite.can_search,
    )
    new_state["stop_reason"] = "awaiting query confirmation"
    return new_state
