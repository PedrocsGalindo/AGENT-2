"""Query planning powered by the local model."""

from __future__ import annotations

from dataclasses import dataclass

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import RankedPaper
from academic_explorer_mvp.llm.local_model import LocalModel, LocalModelError
from academic_explorer_mvp.llm.prompts import (
    build_continue_decision_prompt,
    build_initial_queries_prompt,
    build_refine_queries_prompt,
)


@dataclass(frozen=True)
class ContinueDecision:
    """Model suggestion for continuing the search."""

    should_continue: bool
    reason: str


class QueryPlanner:
    """Only service that asks the local model for search strategy."""

    def __init__(self, model: LocalModel, max_queries_per_round: int = 3) -> None:
        self.model = model
        self.max_queries_per_round = max_queries_per_round

    def plan_initial_queries(self, context: SearchContext) -> list[str]:
        """Ask the local model for initial academic search queries."""

        payload = self._generate_json("initial query planning", build_initial_queries_prompt(context))
        queries = self._extract_queries(payload)
        if not queries:
            raise RuntimeError(
                "Local model returned JSON, but it did not contain a non-empty "
                "`queries` list for initial query planning."
            )
        return queries

    def refine_queries(
        self,
        context: SearchContext,
        ranked_papers: list[RankedPaper],
        used_queries: list[str],
    ) -> list[str]:
        """Ask the local model for next-round queries."""

        payload = self._generate_json(
            "query refinement",
            build_refine_queries_prompt(
                context=context,
                ranked_papers=ranked_papers[:5],
                used_queries=used_queries,
            )
        )
        queries = self._extract_queries(payload)
        if not queries:
            raise RuntimeError(
                "Local model returned JSON, but it did not contain a non-empty "
                "`queries` list for query refinement."
            )
        return queries

    def should_continue(
        self,
        context: SearchContext,
        round_number: int,
        ranked_papers: list[RankedPaper],
        last_new_paper_count: int,
        last_new_useful_count: int,
    ) -> ContinueDecision:
        """Ask the local model whether another round is worth trying."""

        payload = self._generate_json(
            "continue decision",
            build_continue_decision_prompt(
                context=context,
                round_number=round_number,
                ranked_papers=ranked_papers[:5],
                last_new_paper_count=last_new_paper_count,
                last_new_useful_count=last_new_useful_count,
            )
        )
        if not isinstance(payload, dict) or "continue" not in payload:
            raise RuntimeError(
                "Local model returned JSON, but it did not contain the required "
                "`continue` boolean for the continue decision."
            )

        return ContinueDecision(
            should_continue=bool(payload.get("continue")),
            reason=str(payload.get("reason") or "model did not provide a reason"),
        )

    def _generate_json(self, step_name: str, prompt: str) -> dict[str, object] | list[object]:
        try:
            return self.model.generate_json(prompt)
        except LocalModelError as exc:
            raise RuntimeError(
                f"Local model failed during {step_name}: {exc} "
                "There is no deterministic fallback; fix the local model setup or prompt."
            ) from exc

    def _extract_queries(self, payload: object) -> list[str]:
        if isinstance(payload, dict):
            raw_queries = payload.get("queries")
        elif isinstance(payload, list):
            raw_queries = payload
        else:
            raw_queries = None

        if not isinstance(raw_queries, list):
            return []

        queries: list[str] = []
        seen: set[str] = set()
        for item in raw_queries:
            if not isinstance(item, str):
                continue
            query = " ".join(item.split())
            key = query.lower()
            if not query or key in seen:
                continue
            seen.add(key)
            queries.append(query[:160])
            if len(queries) >= self.max_queries_per_round:
                break
        return queries
