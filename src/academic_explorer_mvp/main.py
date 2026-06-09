"""CLI entry point for the Academic Explorer MVP."""

import argparse
import sys

from academic_explorer_mvp.config import load_config
from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph.workflow import run_interactive_graph


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="academic-explorer-mvp",
        description="MVP academico para busca profunda de artigos com LangGraph.",
    )
    parser.add_argument("--query", required=True, help="Tema ou pergunta de pesquisa.")
    parser.add_argument("--min-year", type=int, default=2020, help="Ano minimo dos artigos.")
    parser.add_argument("--max-rounds", type=int, default=2, help="Numero maximo de rodadas.")
    parser.add_argument("--limit", type=int, default=10, help="Limite por provider e por query.")
    return parser


def main() -> None:
    """Run the MVP from the command line."""

    args = build_parser().parse_args()
    context = SearchContext(
        user_query=args.query,
        min_year=args.min_year,
        max_rounds=max(1, args.max_rounds),
        limit=max(1, args.limit),
    )

    try:
        final_state = run_interactive_graph(context=context, config=load_config())
    except RuntimeError as exc:
        print(f"[erro] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_summary(final_state)


def print_summary(state: SearchState) -> None:
    """Print the final search summary."""

    relevant = state.get("relevant_papers", [])
    print("\nAcademic Explorer MVP")
    print("=====================")
    print(f"Rodada atual: {state.get('round_number', 0)}")
    print(f"Motivo de parada: {state.get('stop_reason') or 'sem motivo informado'}")
    if state.get("model_continue_reason"):
        print(f"Decisao do modelo: {state['model_continue_reason']}")

    print("\nQueries usadas:")
    for index, queries in enumerate(state.get("query_history", []), start=1):
        print(f"  Rodada {index}:")
        for query in queries:
            print(f"    - {query}")

    print("\nContagens:")
    print(f"  Resultados brutos: {len(state.get('all_raw_results', []))}")
    print(f"  Artigos normalizados: {len(state.get('normalized_papers', []))}")
    print(f"  Apos deduplicacao: {len(state.get('deduplicated_papers', []))}")
    print(f"  Artigos validados como relevantes: {len(state.get('relevant_papers', []))}")
    print(f"  Artigos excluidos pela validacao: {len(state.get('excluded_papers', []))}")
    print(f"  Novos artigos na ultima rodada: {state.get('last_new_paper_count', 0)}")
    print(f"  Novos artigos uteis na ultima rodada: {state.get('last_new_useful_count', 0)}")
    if state.get("validation_summary"):
        print(f"  Resumo da validacao: {state['validation_summary']}")

    errors = state.get("provider_errors", [])
    if errors:
        print("\nErros de providers:")
        for error in errors:
            print(f"  - {error}")

    print("\nTop artigos validados:")
    if not relevant:
        print("  Nenhum artigo validado como relevante.")
        return

    for position, item in enumerate(relevant[:10], start=1):
        paper = item.get("paper")
        if paper is None:
            continue
        print(f"\n{position}. {paper.title}")
        print(f"   Relevancia: {_relevance_label(str(item.get('relevance') or ''))}")
        print(f"   Ano: {paper.year or 'desconhecido'}")
        print(f"   Fonte: {paper.source}")
        print(f"   URL: {paper.url or 'sem URL'}")
        print(f"   Por que entrou: {item.get('relevance_reason') or 'sem justificativa'}")
        useful_for = item.get("useful_for")
        if useful_for:
            print(f"   Util para: {useful_for}")


def _relevance_label(relevance: str) -> str:
    labels = {
        "high": "alta",
        "medium": "media",
        "low": "baixa",
        "reject": "rejeitada",
    }
    return labels.get(relevance, relevance or "desconhecida")


if __name__ == "__main__":
    main()
