"""Human feedback nodes for ranked papers."""

from academic_explorer_mvp.domain.paper import RankedPaper
from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph.nodes.context_nodes import (
    _paper_feedback,
    _state_text,
    set_paper_feedback,
)


def ask_paper_feedback(state: SearchState) -> SearchState:
    """Ask the user whether the ranked papers are going in the right direction."""

    feedback = _paper_feedback(state)
    new_state = set_paper_feedback(
        state,
        stage="awaiting_paper_feedback",
        pending_answer=None,
        status=None,
        message=_build_feedback_message(state.get("ranked_papers", [])),
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

    restriction = describe_paper_feedback_restriction(answer)
    new_state = set_paper_feedback(
        state,
        stage="feedback_applied",
        answer=answer,
        pending_answer=None,
        status="revised",
        restriction=restriction,
        message=(
            "Entendi. Vou refinar a proxima busca com esta restricao:\n"
            f'"{restriction}"'
        ),
    )
    new_state["stop_reason"] = "paper feedback applied"
    return new_state


def route_after_paper_feedback(state: SearchState) -> str:
    """Route after paper feedback has been interpreted."""

    status = _paper_feedback(state).get("status")
    if status == "accepted":
        return "finalize"
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


def describe_paper_feedback_restriction(user_message: str) -> str:
    """Turn a user critique into a short restriction for the next query round."""

    text = _state_text(user_message)
    lower = text.lower()
    audio_only_markers = (
        "audio-only",
        "audio only",
        "apenas audio",
        "apenas \u00e1udio",
        "somente audio",
        "somente \u00e1udio",
        "so audio",
        "s\u00f3 \u00e1udio",
        "sem audio-visual",
        "sem audiovisual",
        "sem video",
        "sem v\u00eddeo",
        "sem visual",
    )
    if any(marker in lower for marker in audio_only_markers):
        return (
            "audio-only; excluir audio-visual, audiovisual, video, visual, "
            "image, multimodal, text-based detection e hate speech"
        )
    return text


def _build_feedback_message(ranked_papers: list[RankedPaper]) -> str:
    lines = ["Top 10 artigos encontrados pelo score:"]

    if not ranked_papers:
        lines.append("  Nenhum artigo ranqueado.")
    for position, item in enumerate(ranked_papers[:10], start=1):
        paper = item.paper
        year = paper.year or "ano desconhecido"
        source = paper.source or "fonte desconhecida"
        url = paper.url or "sem URL"
        lines.extend(
            [
                f"{position}. {paper.title}",
                f"   Score: {item.score} | Ano: {year} | Fonte: {source}",
                f"   URL: {url}",
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
