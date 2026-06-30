"""CLI entry point for the Academic Explorer MVP."""

import argparse
import sys

from academic_explorer_mvp.config import load_config
from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph.workflow import run_interactive_graph
from academic_explorer_mvp.utils.text import clean_text
from academic_explorer_mvp.utils.validation import relevance_label, validation_counts_from_state


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

    _print_search_filters(state)
    _print_validation_summary(state)

    print("\nContagens:")
    print(f"  Resultados brutos: {len(state.get('all_raw_results', []))}")
    print(f"  Artigos normalizados: {len(state.get('normalized_papers', []))}")
    print(f"  Apos deduplicacao: {len(state.get('deduplicated_papers', []))}")
    print(f"  Artigos validados como relevantes: {len(state.get('relevant_papers', []))}")
    print(f"  Artigos excluidos pela validacao: {len(state.get('excluded_papers', []))}")
    print(f"  Novos artigos na ultima rodada: {state.get('last_new_paper_count', 0)}")
    print(f"  Novos artigos uteis na ultima rodada: {state.get('last_new_useful_count', 0)}")

    errors = state.get("provider_errors", [])
    if errors:
        print("\nErros de providers:")
        for error in errors:
            print(f"  - {error}")

    if not relevant:
        print("  Nenhum artigo validado como relevante.")
        return

    for position, item in enumerate(relevant[:10], start=1):
        paper = item.get("paper")
        if paper is None:
            continue
        print(f"\n{position}. {paper.title}")
        print(f"   Relevancia: {relevance_label(str(item.get('relevance') or ''))}")
        print(f"   Ano: {paper.year or 'desconhecido'}")
        print(f"   Fonte: {paper.source}")
        print(f"   URL: {paper.url or 'sem URL'}")
        print(f"   Por que entrou: {item.get('relevance_reason') or 'sem justificativa'}")
        useful_for = item.get("useful_for")
        if useful_for:
            print(f"   Util para: {useful_for}")


def _print_filter_list(label: str, values: object) -> None:
    items = []
    if isinstance(values, list):
        items = [clean_text(value) for value in values]
        items = [value for value in items if value]

    if not items:
        print(f"  {label}:")
        print("    nenhum")
        print()
        return

    print(f"  {label}:")
    for value in items:
        print(f"    - {value}")
    print()


def _print_search_filters(state: SearchState) -> None:
    search_filters = state.get("search_filters", {}) or {}
    if not isinstance(search_filters, dict):
        search_filters = {}

    print("\nCritérios de validação usados:")
    primary_intent = clean_text(search_filters.get("primary_intent")) or "nenhum"
    print("  Intenção principal:")
    print(f"    {primary_intent}")
    print()

    _print_filter_list("Filtros conservadores", search_filters.get("conservative_filters", []))
    _print_filter_list("Filtros expansivos", search_filters.get("expansive_filters", []))
    _print_filter_list("Restrições negativas", search_filters.get("negative_constraints", []))
    _print_filter_list("Não inferido", search_filters.get("not_inferred", []))
    _print_filter_list("Prioridade da validação", search_filters.get("validation_priority", []))


def _print_validation_summary(state: SearchState) -> None:
    counts = validation_counts_from_state(state)
    print("\nResumo da validação:")
    print(f"  Total avaliados: {counts['total']}")
    print(f"  Incluídos: {counts['included']}")
    print(f"  Excluídos: {counts['excluded']}")
    print(f"  Novos artigos úteis na última rodada: {counts['new_useful']}")
    print("\nDistribuição por relevância:")
    print(f"  Alta: {counts['high']}")
    print(f"  Média: {counts['medium']}")
    print(f"  Baixa: {counts['low']}")
    print(f"  Rejeitada: {counts['reject']}")



if __name__ == "__main__":
    main()
