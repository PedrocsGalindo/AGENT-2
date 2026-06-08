"""Search continuation decision nodes."""

from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.services.query_planner import QueryPlanner
from academic_explorer_mvp.services.ranker import PaperRanker
from academic_explorer_mvp.graph.nodes.context_nodes import _context

def decide_next_step(state: SearchState, planner: QueryPlanner, ranker: PaperRanker) -> SearchState:
    """Validate whether the graph should continue."""

    context = _context(state)
    ranked = state.get("ranked_papers", [])
    good_papers = [item for item in ranked if item.score >= ranker.good_score_threshold]
    min_good_papers = 10
    new_state: SearchState = dict(state)
    if state.get("round_number", 0) >= context.max_rounds:
        new_state["stop_reason"] = "max rounds reached"
        return new_state
    if len(good_papers) >= min_good_papers:
        new_state["stop_reason"] = f"found at least {min_good_papers} good scored papers"
        return new_state
    if state.get("last_new_useful_count", 0) == 0:
        new_state["stop_reason"] = "last round did not bring useful new papers"
        return new_state

    decision = planner.should_continue(
        context=context,
        round_number=state.get("round_number", 0),
        ranked_papers=ranked,
        last_new_paper_count=state.get("last_new_paper_count", 0),
        last_new_useful_count=state.get("last_new_useful_count", 0),
    )
    new_state["model_continue_reason"] = decision.reason
    if not decision.should_continue:
        new_state["stop_reason"] = f"model stopped: {decision.reason}"
    return new_state
