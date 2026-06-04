"""Query planning powered by the local model."""

from dataclasses import dataclass

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import RankedPaper
from academic_explorer_mvp.llm.local_model import LocalModel, LocalModelError
from academic_explorer_mvp.llm.prompts import (
    PromptSpec,
    build_continue_decision_prompt,
    build_enrich_query_prompt,
    build_initial_queries_prompt,
    build_refine_queries_prompt,
    build_rewrite_from_user_revision_prompt,
    build_rewrite_user_query_prompt,
)


@dataclass(frozen=True)
class QueryAssessment:
    """Model assessment of whether the original topic is searchable."""

    can_search: bool
    question: str | None
    reason: str


@dataclass(frozen=True)
class QueryRewrite:
    """Model proposal for one clarified user topic."""

    proposed_query: str
    reason: str


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

    def assess_initial_query(self, context: SearchContext) -> QueryAssessment:
        """Ask the local model if the user's topic is ready for search."""

        payload = self._expect_dict(
            "initial query assessment",
            self._generate_json(
                "initial query assessment",
                build_enrich_query_prompt(context),
            ),
        )

        can_search = self._required_bool(payload, "can_search", "initial query assessment")
        question = self._optional_string(payload, "question")
        reason = self._optional_string(payload, "reason") or "model did not provide a reason"

        if not can_search and not question:
            raise RuntimeError(
                "Initial query assessment requires `question` when `can_search` is false."
            )

        return QueryAssessment(
            can_search=can_search,
            question=None if can_search else question,
            reason=reason,
        )

    def rewrite_user_query(
        self,
        initial_query: str,
        question: str,
        question_reason: str,
        user_answer: str,
    ) -> QueryRewrite:
        """Ask the local model for one refined topic after clarification."""

        payload = self._expect_dict(
            "clarified user query rewriting",
            self._generate_json(
                "clarified user query rewriting",
                build_rewrite_user_query_prompt(
                    initial_query=initial_query,
                    question=question,
                    question_reason=question_reason,
                    user_answer=user_answer,
                ),
            ),
        )
        proposed_query = self._required_string(
            payload,
            "proposed_query",
            "clarified user query rewriting",
        )

        return QueryRewrite(
            proposed_query=proposed_query[:240],
            reason=self._optional_string(payload, "reason") or "model did not provide a reason",
        )

    def rewrite_from_user_revision(
        self,
        initial_query: str,
        proposed_query: str,
        clarification_question: str,
        clarification_reason: str,
        clarification_answer: str,
        user_revision: str,
    ) -> QueryRewrite:
        """Ask the local model to normalize a user's revised topic."""

        payload = self._expect_dict(
            "user revision query rewriting",
            self._generate_json(
                "user revision query rewriting",
                build_rewrite_from_user_revision_prompt(
                    initial_query=initial_query,
                    proposed_query=proposed_query,
                    clarification_question=clarification_question,
                    clarification_reason=clarification_reason,
                    clarification_answer=clarification_answer,
                    user_revision=user_revision,
                ),
            ),
        )
        revised_query = self._required_string(
            payload,
            "proposed_query",
            "user revision query rewriting",
        )

        return QueryRewrite(
            proposed_query=revised_query[:240],
            reason=self._optional_string(payload, "reason") or "model did not provide a reason",
        )

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
            ),
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
            ),
        )
        payload = self._expect_dict("continue decision", payload)

        return ContinueDecision(
            should_continue=self._required_bool(payload, "continue", "continue decision"),
            reason=self._optional_string(payload, "reason") or "model did not provide a reason",
        )

    def _generate_json(self, step_name: str, prompt: PromptSpec | str) -> dict[str, object] | list[object]:
        try:
            prompt_text = prompt.text if isinstance(prompt, PromptSpec) else prompt
            return self.model.generate_json(prompt_text)
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

    def _extract_bool(
        self,
        payload: dict[str, object],
        key: str,
        default: bool | None = None,
    ) -> bool:
        value = payload.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text == "true":
                return True
            if text == "false":
                return False
        if default is not None:
            return default
        raise RuntimeError(
            "Local model returned JSON, but it did not contain the required "
            f"`{key}` boolean."
        )

    def _expect_dict(self, step_name: str, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Local model returned JSON, but it did not contain an object for {step_name}."
            )
        return payload

    def _required_string(
        self,
        payload: dict[str, object],
        key: str,
        step_name: str,
    ) -> str:
        value = self._safe_string(payload, key)
        if not value:
            raise RuntimeError(
                "Local model returned JSON, but it did not contain the required "
                f"`{key}` string for {step_name}."
            )
        return value

    def _optional_string(
        self,
        payload: dict[str, object],
        key: str,
        default: str | None = None,
    ) -> str | None:
        value = payload.get(key)
        if value is None:
            return default
        text = " ".join(str(value).split())
        return text or default

    def _required_bool(
        self,
        payload: dict[str, object],
        key: str,
        step_name: str,
    ) -> bool:
        try:
            return self._extract_bool(payload, key)
        except RuntimeError as exc:
            raise RuntimeError(
                "Local model returned JSON, but it did not contain the required "
                f"`{key}` boolean for {step_name}."
            ) from exc

    def _safe_string(
        self,
        payload: dict[str, object],
        key: str,
        default: str = "",
    ) -> str:
        value = payload.get(key)
        if value is None:
            value = default
        return " ".join(str(value).split())
