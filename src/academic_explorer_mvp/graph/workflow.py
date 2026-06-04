"""Build and run the LangGraph search workflow."""

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
    graph.add_node("assess_initial_query", lambda state: nodes.assess_initial_query(state, planner))
    graph.add_node(
        "rewrite_user_query_after_clarification",
        lambda state: nodes.rewrite_user_query_after_clarification(state, planner),
    )
    graph.add_node(
        "handle_query_confirmation_or_revision",
        lambda state: nodes.handle_query_confirmation_or_revision(state, planner),
    )
    graph.add_node("commit_enriched_query", nodes.commit_enriched_query)
    graph.add_node("plan_queries", lambda state: nodes.plan_queries(state, planner))
    graph.add_node("search_papers", lambda state: nodes.search_papers(state, search_service))
    graph.add_node("normalize_papers", lambda state: nodes.normalize_papers(state, normalizer))
    graph.add_node("deduplicate_papers", lambda state: nodes.deduplicate_papers(state, deduplicator))
    graph.add_node("rank_papers", lambda state: nodes.rank_papers(state, ranker))
    graph.add_node("decide_next_step", lambda state: nodes.decide_next_step(state, planner, ranker))
    graph.add_node("finalize", nodes.finalize)

    graph.set_entry_point("initialize_context")
    graph.add_conditional_edges(
        "initialize_context",
        nodes.route_after_context_initialization,
        {
            "assess_initial_query": "assess_initial_query",
            "rewrite_user_query_after_clarification": "rewrite_user_query_after_clarification",
            "handle_query_confirmation_or_revision": "handle_query_confirmation_or_revision",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "assess_initial_query",
        nodes.route_after_initial_assessment,
        {
            "ready_to_search": "plan_queries",
            "needs_clarification": "finalize",
        },
    )
    graph.add_edge("rewrite_user_query_after_clarification", "finalize")
    graph.add_conditional_edges(
        "handle_query_confirmation_or_revision",
        nodes.route_after_query_confirmation,
        {
            "commit_query": "commit_enriched_query",
            "wait_for_user": "finalize",
        },
    )
    graph.add_edge("commit_enriched_query", "assess_initial_query")

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


## INTERACTION WITH USER 

def run_interactive_graph(context: SearchContext, config: AppConfig) -> SearchState:
    """Run the workflow interactively from the CLI."""

    graph = build_graph(config)
    state: SearchState = {"context": context}

    while True:
        state = graph.invoke(state)

        enrichment = state.get("query_enrichment", {}) or {}
        stage = enrichment.get("stage")

        if stage == "awaiting_clarification_answer":
            question = enrichment.get("question")
            reason = enrichment.get("reason")

            print("\nPreciso de uma clarificação antes de buscar.")
            if question:
                print(f"\nPergunta: {question}")
            if reason:
                print(f"Motivo: {reason}")
            answer = input("\nSua resposta: ").strip()
            while not answer:
                answer = input("Digite uma resposta: ").strip()

            state = _update_query_enrichment(
                state,
                answer=answer,
            )

            state["stop_reason"] = None
            continue

        if stage in {"awaiting_query_confirmation", "unclear_confirmation"}:
            message = enrichment.get("message")

            if message:
                print("\n" + message)
            else:
                print('\nResponda "sim" para aceitar ou escreva uma versão melhor.')

            answer = input("\nSua resposta: ").strip()

            while not answer:
                answer = input('Responda "sim" ou escreva uma versão melhor: ').strip()

            state = _update_query_enrichment(
                state,
                confirmation_answer=answer,
            )

            state["stop_reason"] = None
            continue

        return state
    
def _update_query_enrichment(state: SearchState, **updates: object) -> SearchState:
    """Return a copied state with updated query enrichment data."""

    new_state: SearchState = dict(state)

    enrichment = dict(new_state.get("query_enrichment", {}) or {})
    enrichment.update(updates)

    new_state["query_enrichment"] = enrichment
    return new_state
