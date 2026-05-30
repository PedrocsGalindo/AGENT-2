"""Build and run the LangGraph search workflow."""

from __future__ import annotations

from academic_explorer_mvp.config import AppConfig
from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph import nodes
from academic_explorer_mvp.graph.routers import route_after_decision
from academic_explorer_mvp.llm.local_model import LocalModel
from academic_explorer_mvp.providers.openalex import OpenAlexProvider
from academic_explorer_mvp.providers.semantic_scholar import SemanticScholarProvider
from academic_explorer_mvp.services.deduplicator import PaperDeduplicator
from academic_explorer_mvp.services.normalizer import PaperNormalizer
from academic_explorer_mvp.services.query_planner import QueryPlanner
from academic_explorer_mvp.services.ranker import PaperRanker
from academic_explorer_mvp.services.search_service import SearchService


def build_graph(config: AppConfig):
    """Build the LangGraph workflow with simple service instances."""

    try:
        from langgraph.graph import END, StateGraph
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "langgraph is required for this MVP. Install base dependencies with: "
            "py -m pip install -e ."
        ) from exc

    local_model = LocalModel(config)
    planner = QueryPlanner(local_model)
    search_service = SearchService(
        providers=[
            OpenAlexProvider(
                mailto=config.openalex_mailto,
                timeout_seconds=config.provider_timeout_seconds,
            ),
            SemanticScholarProvider(
                api_key=config.semantic_scholar_api_key,
                timeout_seconds=config.provider_timeout_seconds,
            ),
        ]
    )
    normalizer = PaperNormalizer()
    deduplicator = PaperDeduplicator()
    ranker = PaperRanker()

    graph = StateGraph(SearchState)
    graph.add_node("initialize_context", nodes.initialize_context)
    graph.add_node("plan_queries", lambda state: nodes.plan_queries(state, planner))
    graph.add_node("search_papers", lambda state: nodes.search_papers(state, search_service))
    graph.add_node("normalize_papers", lambda state: nodes.normalize_papers(state, normalizer))
    graph.add_node("deduplicate_papers", lambda state: nodes.deduplicate_papers(state, deduplicator))
    graph.add_node("rank_papers", lambda state: nodes.rank_papers(state, ranker))
    graph.add_node("decide_next_step", lambda state: nodes.decide_next_step(state, planner, ranker))
    graph.add_node("finalize", nodes.finalize)

    graph.set_entry_point("initialize_context")
    graph.add_edge("initialize_context", "plan_queries")
    graph.add_edge("plan_queries", "search_papers")
    graph.add_edge("search_papers", "normalize_papers")
    graph.add_edge("normalize_papers", "deduplicate_papers")
    graph.add_edge("deduplicate_papers", "rank_papers")
    graph.add_edge("rank_papers", "decide_next_step")
    graph.add_conditional_edges(
        "decide_next_step",
        route_after_decision,
        {
            "continue": "plan_queries",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_graph(context: SearchContext, config: AppConfig) -> SearchState:
    """Run the workflow for one CLI request."""

    graph = build_graph(config)
    initial_state: SearchState = {"context": context}
    return graph.invoke(initial_state)
