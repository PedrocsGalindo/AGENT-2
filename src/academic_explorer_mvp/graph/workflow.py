"""Build and run the LangGraph search workflow."""

from academic_explorer_mvp.config import AppConfig
from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph import nodes
from academic_explorer_mvp.llm.local_model import LocalModel
from academic_explorer_mvp.providers.openalex import OpenAlexProvider
from academic_explorer_mvp.providers.semantic_scholar import SemanticScholarProvider
from academic_explorer_mvp.services.deduplicator import PaperDeduplicator
from academic_explorer_mvp.services.normalizer import PaperNormalizer
from academic_explorer_mvp.services.query_planner import QueryPlanner
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
    graph = StateGraph(SearchState)
    ## INITIAL QUERY
    graph.add_node("initialize_context", nodes.initialize_context)
    graph.add_node("enough_context_query", lambda state: nodes.enough_context_query(state, planner))
    graph.add_node("ask_context_question", lambda state: nodes.ask_context_question(state, planner))
    graph.add_node(
        "rewrite_user_query_after_clarification",
        lambda state: nodes.rewrite_user_query_after_clarification(state, planner),
    )
    graph.add_node(
        "handle_query_confirmation_or_revision",
        lambda state: nodes.handle_query_confirmation_or_revision(state, planner),
    )
    graph.add_node("commit_enriched_query", nodes.commit_enriched_query)
    ## SEARCH 
    graph.add_node("plan_filters", lambda state: nodes.plan_filters(state, planner))
    graph.add_node("plan_queries", lambda state: nodes.plan_queries(state, planner))
    graph.add_node("search_papers", lambda state: nodes.search_papers(state, search_service))
    graph.add_node("normalize_papers", lambda state: nodes.normalize_papers(state, normalizer))
    graph.add_node("deduplicate_papers", lambda state: nodes.deduplicate_papers(state, deduplicator))
    graph.add_node("validate_papers", lambda state: nodes.validate_papers(state, planner))
    graph.add_node(
        "judge_paper_validations",
        lambda state: nodes.judge_paper_validations(state, planner),
    )
    graph.add_node("ask_paper_feedback", nodes.ask_paper_feedback)
    graph.add_node("handle_paper_feedback", nodes.handle_paper_feedback)
    graph.add_node("analyze_search_feedback", lambda state: nodes.analyze_search_feedback(state, planner))
    graph.add_node("decide_next_step", lambda state: nodes.decide_next_step(state, planner))
    ##
    graph.add_node("finalize", lambda state: state)
    graph.add_node("wait_for_user", lambda state: state)
    
    # EDGES 
    graph.set_entry_point("initialize_context")
    graph.add_conditional_edges(
        "initialize_context",
        nodes.route_after_context_initialization,
        {
            "enough_context_query": "enough_context_query",
            "rewrite_user_query_after_clarification": "rewrite_user_query_after_clarification",
            "handle_query_confirmation_or_revision": "handle_query_confirmation_or_revision",
            "handle_paper_feedback": "handle_paper_feedback",
            "analyze_search_feedback": "analyze_search_feedback",
            "plan_filters": "plan_filters",
            "plan_queries": "plan_queries",
            "search_papers": "search_papers",
            "wait_for_user": "wait_for_user",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "enough_context_query",
        nodes.route_after_initial_assessment,
        {
            "ready_to_search": "plan_filters",
            "needs_clarification": "ask_context_question",
        },
    )
    graph.add_edge("ask_context_question", "wait_for_user")
    graph.add_edge("rewrite_user_query_after_clarification", "wait_for_user")
    graph.add_conditional_edges(
        "handle_query_confirmation_or_revision",
        nodes.route_after_query_confirmation,
        {
            "commit_query": "commit_enriched_query",
            "wait_for_user": "wait_for_user",
        },
    )
    graph.add_edge("commit_enriched_query", "plan_filters")

    graph.add_edge("plan_filters", "plan_queries")
    graph.add_edge("plan_queries", "wait_for_user")
    graph.add_edge("search_papers", "normalize_papers")
    graph.add_edge("normalize_papers", "deduplicate_papers")
    graph.add_edge("deduplicate_papers", "validate_papers")
    graph.add_edge("validate_papers", "judge_paper_validations")
    graph.add_edge("judge_paper_validations", "decide_next_step")
    graph.add_edge("decide_next_step", "ask_paper_feedback")
    graph.add_edge("ask_paper_feedback", "wait_for_user")
    graph.add_conditional_edges(
        "handle_paper_feedback",
        nodes.route_after_paper_feedback,
        {
            "wait_for_user": "wait_for_user",
            "analyze_search_feedback": "analyze_search_feedback",
            "finalize": "finalize",
        },
    )
    graph.add_edge("analyze_search_feedback", "plan_filters")
    graph.add_edge("wait_for_user", END)
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

        preview = state.get("query_preview", {}) or {}
        preview_stage = preview.get("stage")

        if preview_stage == "awaiting_query_preview":
            queries = state.get("pending_queries", [])
            round_number = preview.get("round") or state.get("round_number", 0) + 1

            if stage == "ready_to_search" and state.get("round_number", 0) == 0:
                print("\nOk, compreendi o que voce deseja procurar.")

            print(f"\nQueries que serao usadas na rodada {round_number}:")
            if queries:
                for query in queries:
                    print(f"  - {query}")
            else:
                print("  Nenhuma query planejada.")

            state = _update_query_preview(
                state,
                stage="shown",
            )
            state["stop_reason"] = None
            continue

        feedback = state.get("paper_feedback", {}) or {}
        feedback_stage = feedback.get("stage")

        if feedback_stage in {"awaiting_paper_feedback", "unclear_paper_feedback"}:
            message = feedback.get("message")

            if message:
                print("\n" + str(message))
            else:
                print(
                    '\nResponda "sim" para aceitar a direcao da busca ou '
                    "escreva uma critica/direcionamento."
                )

            answer = input("\nSua resposta: ").strip()

            while not answer:
                answer = input(
                    'Responda "sim" ou escreva uma critica/direcionamento: '
                ).strip()

            state = _update_paper_feedback(
                state,
                pending_answer=answer,
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


def _update_query_preview(state: SearchState, **updates: object) -> SearchState:
    """Return a copied state with updated query preview data."""

    new_state: SearchState = dict(state)

    preview = dict(new_state.get("query_preview", {}) or {})
    preview.update(updates)

    new_state["query_preview"] = preview
    return new_state


def _update_paper_feedback(state: SearchState, **updates: object) -> SearchState:
    """Return a copied state with updated paper feedback data."""

    new_state: SearchState = dict(state)

    feedback = dict(new_state.get("paper_feedback", {}) or {})
    feedback.update(updates)

    new_state["paper_feedback"] = feedback
    return new_state
