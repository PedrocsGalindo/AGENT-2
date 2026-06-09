"""Nodes for the initial query clarification flow."""

from dataclasses import replace

from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph.nodes.context_nodes import (
    PAUSE_STOP_REASONS,
    _context,
    _paper_feedback,
    _query_preview,
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
    preview = _query_preview(state)
    preview_stage = preview.get("stage")
    feedback = _paper_feedback(state)
    feedback_stage = feedback.get("stage")

    if preview_stage == "shown" and state.get("pending_queries"):
        return "search_papers"

    if preview_stage == "awaiting_query_preview":
        return "wait_for_user"

    if feedback_stage == "accepted":
        return "finalize"

    if feedback_stage == "needs_feedback_analysis":
        return "analyze_search_feedback"

    if feedback_stage == "feedback_analyzed":
        return "plan_filters"

    if feedback_stage in {"awaiting_paper_feedback", "unclear_paper_feedback"}:
        if _state_text(feedback.get("pending_answer")):
            return "handle_paper_feedback"
        return "wait_for_user"

    if _state_text(feedback.get("pending_answer")):
        return "handle_paper_feedback"

    if stage == "awaiting_clarification_answer":
        if _state_text(enrichment.get("answer")):
            return "rewrite_user_query_after_clarification"
        return "wait_for_user"

    if stage in {"awaiting_query_confirmation", "unclear_confirmation"}:
        if _state_text(enrichment.get("confirmation_answer")):
            return "handle_query_confirmation_or_revision"
        return "wait_for_user"

    if _state_text(enrichment.get("confirmation_answer")):
        return "handle_query_confirmation_or_revision"

    if _state_text(enrichment.get("answer")) and enrichment.get("question"):
        return "rewrite_user_query_after_clarification"

    if stage == "ready_to_search":
        return "plan_filters"

    return "enough_context_query"


def enough_context_query(state: SearchState, planner: QueryPlanner) -> SearchState:
    """Assess whether the initial user topic has enough search context."""

    context = _context(state)
    enrichment = _query_enrichment(state)
    assessment = planner.assess_query_context(context)
    original_query = _state_text(enrichment.get("original_query")) or context.user_query

    if assessment.has_enough_context:
        new_state = set_query_enrichment(
            state,
            stage="ready_to_search",
            original_query=original_query,
            has_enough_context=True,
            reason=assessment.reason,
        )
        if new_state.get("stop_reason") in PAUSE_STOP_REASONS:
            new_state["stop_reason"] = None
        return new_state

    new_state = set_query_enrichment(
        state,
        stage="needs_clarification",
        original_query=original_query,
        has_enough_context=False,
        reason=assessment.reason,
        question=None,
        answer=None,
        proposed_query=None,
        confirmation_answer=None,
        confirmation_status=None,
        user_revision=None,
        message=None,
        round=int(enrichment.get("round") or 0) + 1,
    )
    return new_state


def route_after_initial_assessment(state: SearchState) -> str:
    stage = _query_enrichment(state).get("stage")

    if stage == "ready_to_search":
        return "ready_to_search"
    if stage == "needs_clarification":
        return "needs_clarification"
    return "needs_clarification"


def ask_context_question(state: SearchState, planner: QueryPlanner) -> SearchState:
    """Generate one clarification question after context assessment."""

    context = _context(state)
    enrichment = _query_enrichment(state)
    reason = _required_text(enrichment.get("reason"), "reason")

    question = planner.generate_context_question(
        user_query=context.user_query,
        reason=reason,
    )

    new_state = set_query_enrichment(
        state,
        stage="awaiting_clarification_answer",
        question=question.question,
        reason=question.reason,
        answer=None,
    )
    new_state["stop_reason"] = "awaiting clarification answer"
    return new_state


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
    )
    new_state["stop_reason"] = "awaiting query confirmation"
    return new_state
