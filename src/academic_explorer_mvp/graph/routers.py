from academic_explorer_mvp.domain.state import SearchState


def route_after_decision(state: SearchState) -> str:
    """Route to another query-planning round or finalization."""

    if state.get("stop_reason"):
        return "finalize"
    return "continue"
