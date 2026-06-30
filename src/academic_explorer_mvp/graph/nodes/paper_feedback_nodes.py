"""Human feedback nodes for ranked papers."""

from dataclasses import replace

from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph.nodes.context_nodes import (
    _context,
    _paper_feedback,
    _query_enrichment,
    _state_text,
    set_paper_feedback,
    set_search_feedback,
)
from academic_explorer_mvp.services.query_planner import QueryPlanner
from academic_explorer_mvp.utils.validation import relevance_label, validation_counts_from_state


REJECTED_PAPERS_PREVIEW_LIMIT = 10


def ask_paper_feedback(state: SearchState) -> SearchState:
    """Ask the user whether the ranked papers are going in the right direction."""

    feedback = _paper_feedback(state)
    new_state = set_paper_feedback(
        state,
        stage="awaiting_paper_feedback",
        pending_answer=None,
        status=None,
        message=_build_feedback_message(state),
        round=int(feedback.get("round") or 0) + 1,
    )
    new_state["stop_reason"] = "awaiting paper feedback"
    return new_state


def handle_paper_feedback(state: SearchState) -> SearchState:
    """Handle acceptance or a user critique of the ranked-paper direction."""

    feedback = _paper_feedback(state)
    answer = _state_text(feedback.get("pending_answer"))
    status = interpret_paper_feedback(answer)

    if status == "accepted":
        new_state = set_paper_feedback(
            state,
            stage="accepted",
            pending_answer=None,
            status="accepted",
            message=None,
        )
        new_state["stop_reason"] = "user accepted ranked paper direction"
        return new_state

    if status == "unclear":
        new_state = set_paper_feedback(
            state,
            stage="unclear_paper_feedback",
            pending_answer=None,
            status="unclear",
            message=(
                'Responda "sim" para aceitar a direcao da busca ou escreva '
                "uma critica/direcionamento com pelo menos alguns termos."
            ),
        )
        new_state["stop_reason"] = "awaiting paper feedback"
        return new_state

    new_state = set_paper_feedback(
        state,
        stage="needs_feedback_analysis",
        answer=answer,
        pending_answer=None,
        status="revised",
        message=None,
    )
    new_state["stop_reason"] = None
    return new_state


def route_after_paper_feedback(state: SearchState) -> str:
    """Route after paper feedback has been interpreted."""

    status = _paper_feedback(state).get("status")
    if status == "accepted":
        return "finalize"
    if status == "revised":
        return "analyze_search_feedback"
    return "wait_for_user"


def interpret_paper_feedback(user_message: str) -> str:
    """Interpret a paper-feedback answer as accepted, revised, or unclear."""

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


def analyze_search_feedback(state: SearchState, planner: QueryPlanner) -> SearchState:
    """Interpret user feedback before planning another search round."""

    context = _context(state)
    enrichment = _query_enrichment(state)
    feedback = _paper_feedback(state)
    original_query = (
        _state_text(enrichment.get("original_query"))
        or _state_text(enrichment.get("resolved_query"))
        or context.user_query
    )
    refined_query = _state_text(enrichment.get("resolved_query")) or context.user_query
    answer = _state_text(feedback.get("answer"))

    analysis = planner.analyze_search_feedback(
        original_query=original_query,
        refined_query=refined_query,
        user_feedback=answer,
        validated_papers=state.get("validated_papers", [])[:20],
        used_queries=state.get("used_queries", []),
    )

    new_state: SearchState = dict(state)
    new_state["context"] = replace(context, user_query=analysis.revised_topic)
    new_state["pending_queries"] = []
    new_state["raw_results"] = []
    new_state["all_raw_results"] = []
    new_state["normalized_papers"] = []
    new_state["deduplicated_papers"] = []
    new_state["ranked_papers"] = []
    new_state["validated_papers"] = []
    new_state["relevant_papers"] = []
    new_state["excluded_papers"] = []
    new_state["validation_summary"] = None
    new_state["model_validation_summary"] = None
    new_state["judge_validation_summary"] = None
    new_state["judge_corrections_count"] = 0
    new_state["validation_counts"] = {}
    new_state["known_paper_ids"] = []
    new_state["last_new_paper_count"] = 0
    new_state["last_new_paper_ids"] = []
    new_state["last_new_useful_count"] = 0
    new_state["stop_reason"] = None
    new_state = set_search_feedback(
        new_state,
        stage="analyzed",
        revised_topic=analysis.revised_topic,
        positive_constraints=analysis.positive_constraints,
        negative_constraints=analysis.negative_constraints,
        query_strategy=analysis.query_strategy,
        reason=analysis.reason,
    )
    new_state["search_filters"] = {"stage": "idle"}
    new_state = set_paper_feedback(
        new_state,
        stage="feedback_analyzed",
        status="revised",
        pending_answer=None,
        message=None,
    )
    return new_state


def _build_feedback_message(state: SearchState) -> str:
    relevant_papers = state.get("relevant_papers", [])
    excluded_papers = state.get("excluded_papers", [])
    lines = []

    _append_search_filters(lines, state.get("search_filters", {}) or {})
    _append_validation_summary(lines, state)

    if not relevant_papers:
        lines.append("  Nenhum artigo validado como relevante.")
    for position, item in enumerate(relevant_papers, start=1):
        paper = item.get("paper")
        if paper is None:
            continue
        year = getattr(paper, "year", None) or "ano desconhecido"
        source = getattr(paper, "source", None) or "fonte desconhecida"
        url = getattr(paper, "url", None) or "sem URL"
        corrected_relevance = _state_text(item.get("corrected_relevance"))
        relevance = relevance_label(
            corrected_relevance or _state_text(item.get("relevance"))
        )
        use_judge_reason = bool(item.get("judge_correction_applied")) or (
            item.get("reason_is_supported") is False
        )
        reason = (
            _state_text(item.get("judge_reason"))
            if use_judge_reason
            else _state_text(item.get("relevance_reason"))
        )
        reason = reason or _state_text(item.get("relevance_reason"))
        reason = reason or "Sem justificativa informada."
        useful_for = _state_text(item.get("useful_for"))
        lines.extend(
            [
                f"{position}. {getattr(paper, 'title', 'titulo desconhecido')}",
                f"   Relevancia: {relevance} | Ano: {year} | Fonte: {source}",
                f"   URL: {url}",
                f"   Por que entrou: {reason}",
            ]
        )
        if useful_for:
            lines.append(f"   Util para: {useful_for}")

    if excluded_papers:
        rejected_preview = excluded_papers[:REJECTED_PAPERS_PREVIEW_LIMIT]
        lines.extend(
            [
                "",
                "Exemplos rejeitados pela validacao:",
                (
                    f"  Mostrando {len(rejected_preview)} de "
                    f"{len(excluded_papers)} rejeitados."
                ),
                "",
            ]
        )
        for item in rejected_preview:
            paper = item.get("paper")
            title = getattr(paper, "title", "titulo desconhecido")
            corrected_relevance = _state_text(item.get("corrected_relevance"))
            relevance = corrected_relevance or _state_text(item.get("relevance")) or "reject"
            label_title = (
                "Relevancia corrigida" if corrected_relevance else "Relevancia"
            )
            reason = (
                _state_text(item.get("judge_reason"))
                or _state_text(item.get("mismatch_reason"))
                or "fora da intencao da busca"
            )
            lines.extend(
                [
                    f"  - {title}",
                    f"    {label_title}: {relevance_label(relevance)}",
                    f"    Motivo: {reason}",
                    "",
                ]
            )

    lines.extend(
        [
            "",
            "A direcao esta correta?",
            'Responda "sim" para aceitar ou escreva uma critica/direcionamento.',
        ]
    )
    return "\n".join(lines)


def _append_search_filters(lines: list[str], search_filters: dict[str, object]) -> None:
    lines.append("Critérios de validação usados:")
    _append_filter_value(lines, "Intenção principal", search_filters.get("primary_intent"))
    _append_filter_list(lines, "Filtros conservadores", search_filters.get("conservative_filters"))
    _append_filter_list(lines, "Filtros expansivos", search_filters.get("expansive_filters"))
    _append_filter_list(lines, "Restrições negativas", search_filters.get("negative_constraints"))
    _append_filter_list(lines, "Não inferido", search_filters.get("not_inferred"))
    _append_filter_list(lines, "Prioridade da validação", search_filters.get("validation_priority"))
    lines.append("")


def _append_filter_value(lines: list[str], label: str, value: object) -> None:
    text = _state_text(value) or "nenhum"
    lines.extend([f"  {label}:", f"    {text}", ""])


def _append_filter_list(lines: list[str], label: str, values: object) -> None:
    lines.append(f"  {label}:")
    items = []
    if isinstance(values, list):
        items = [_state_text(value) for value in values]
        items = [value for value in items if value]

    if not items:
        lines.append("    nenhum")
    else:
        for value in items:
            lines.append(f"    - {value}")
    lines.append("")


def _append_validation_summary(lines: list[str], state: SearchState) -> None:
    counts = validation_counts_from_state(state)
    lines.extend(
        [
            "Resumo da validação:",
            f"  Total avaliados: {counts['total']}",
            f"  Incluídos: {counts['included']}",
            f"  Excluídos: {counts['excluded']}",
            f"  Novos artigos úteis na última rodada: {counts['new_useful']}",
            "",
            "Distribuição por relevância:",
            f"  Alta: {counts['high']}",
            f"  Média: {counts['medium']}",
            f"  Baixa: {counts['low']}",
            f"  Rejeitada: {counts['reject']}",
            "",
        ]
    )

