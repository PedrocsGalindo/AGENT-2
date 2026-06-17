"""Query planning powered by the local model."""

from dataclasses import dataclass
import re
import unicodedata

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import Paper
from academic_explorer_mvp.llm.local_model import LocalModel, LocalModelError
from academic_explorer_mvp.llm.prompts import (
    PromptSpec,
    build_assess_query_context_prompt,
    build_context_question_prompt,
    build_continue_decision_prompt,
    build_feedback_analysis_prompt,
    build_initial_queries_prompt,
    build_plan_filters_prompt,
    build_refine_queries_prompt,
    build_rewrite_from_user_revision_prompt,
    build_rewrite_user_query_prompt,
    build_validate_papers_prompt,
)

def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]

@dataclass(frozen=True)
class QueryAssessment:
    """Model assessment of whether the original topic is searchable."""

    can_search: bool
    question: str | None
    reason: str


@dataclass(frozen=True)
class QueryContextAssessment:
    """Model assessment of whether the query has enough context."""

    has_enough_context: bool
    reason: str


@dataclass(frozen=True)
class ContextQuestion:
    """Model-generated clarification question."""

    question: str
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


@dataclass(frozen=True)
class SearchFeedbackAnalysis:
    """Structured interpretation of user feedback about ranked papers."""

    revised_topic: str
    positive_constraints: list[str]
    negative_constraints: list[str]
    query_strategy: str
    reason: str


@dataclass(frozen=True)
class SearchFilters:
    """Structured semantic filters for paper validation."""

    primary_intent: str
    conservative_filters: list[str]
    expansive_filters: list[str]
    negative_constraints: list[str]
    not_inferred: list[str]
    validation_priority: list[str]
    reason: str

    def to_state(self) -> dict[str, object]:
        return {
            "primary_intent": self.primary_intent,
            "conservative_filters": self.conservative_filters,
            "expansive_filters": self.expansive_filters,
            "negative_constraints": self.negative_constraints,
            "not_inferred": self.not_inferred,
            "validation_priority": self.validation_priority,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PaperValidation:
    """Model validation for one candidate paper."""

    paper_id: str
    relevance: str
    decision: str
    relevance_reason: str
    mismatch_reason: str
    useful_for: str


@dataclass(frozen=True)
class PaperValidationResult:
    """Semantic validation result for candidate papers."""

    validated_papers: list[PaperValidation]
    summary: str


class QueryPlanner:
    """Only service that asks the local model for search strategy."""

    def __init__(self, model: LocalModel, max_queries_per_round: int = 3) -> None:
        self.model = model
        self.max_queries_per_round = max_queries_per_round

    def assess_query_context(self, context: SearchContext) -> QueryContextAssessment:
        """Ask the local model whether the user's topic has enough context."""

        payload = self._expect_dict(
            "query context assessment",
            self._generate_json(
                "query context assessment",
                build_assess_query_context_prompt(context),
            ),
        )

        has_enough_context = self._required_bool(
            payload,
            "has_enough_context",
            "query context assessment",
        )
        reason = self._optional_string(payload, "reason") or "model did not provide a reason"

        forced_assessment = self._forced_query_context_assessment(context.user_query)
        if forced_assessment is not None:
            return forced_assessment

        return QueryContextAssessment(
            has_enough_context=has_enough_context,
            reason=reason,
        )

    def generate_context_question(
        self,
        user_query: str,
        reason: str,
    ) -> ContextQuestion:
        """Ask the local model for one clarification question with options."""

        payload = self._expect_dict(
            "context question generation",
            self._generate_json(
                "context question generation",
                build_context_question_prompt(
                    user_query=user_query,
                    reason=reason,
                ),
            ),
        )

        question = self._required_string(
            payload,
            "question",
            "context question generation",
        )

        return ContextQuestion(
            question=question,
            reason=self._optional_string(payload, "reason") or "model did not provide a reason",
        )

    def assess_initial_query(self, context: SearchContext) -> QueryAssessment:
        """Compatibility wrapper for the split query-context flow."""

        assessment = self.assess_query_context(context)
        if assessment.has_enough_context:
            return QueryAssessment(
                can_search=True,
                question=None,
                reason=assessment.reason,
            )

        question = self.generate_context_question(
            user_query=context.user_query,
            reason=assessment.reason,
        )
        return QueryAssessment(
            can_search=False,
            question=question.question,
            reason=question.reason,
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
        if len(queries) != self.max_queries_per_round:
            raise RuntimeError(
                "Local model returned JSON, but it did not contain a "
                f"`queries` list with exactly {self.max_queries_per_round} unique "
                "queries for initial query planning."
            )
        return queries

    def plan_filters(self, context: SearchContext) -> SearchFilters:
        """Ask the local model for semantic filters before paper validation."""

        payload = self._expect_dict(
            "semantic filter planning",
            self._generate_json(
                "semantic filter planning",
                build_plan_filters_prompt(context),
            ),
        )
        payload = self._normalize_filter_payload(
            payload,
            fallback_primary_intent=context.user_query,
        )

        return SearchFilters(
            primary_intent=self._required_string(
                payload,
                "primary_intent",
                "semantic filter planning",
            )[:300],
            conservative_filters=self._optional_string_list(
                payload,
                "conservative_filters",
            ),
            expansive_filters=self._optional_string_list(payload, "expansive_filters"),
            negative_constraints=self._optional_string_list(
                payload,
                "negative_constraints",
                max_length=300,
            ),
            not_inferred=self._optional_string_list(payload, "not_inferred"),
            validation_priority=self._optional_string_list(payload, "validation_priority"),
            reason=(
                self._optional_string(payload, "reason")
                or "model did not provide a reason"
            )[:500],
        )

    def refine_queries(
        self,
        context: SearchContext,
        validated_papers: list[dict[str, object]],
        used_queries: list[str],
        search_feedback: dict[str, object] | None = None,
    ) -> list[str]:
        """Ask the local model for next-round queries."""

        payload = self._generate_json(
            "query refinement",
            build_refine_queries_prompt(
                context=context,
                validated_papers=validated_papers[:10],
                used_queries=used_queries,
                search_feedback=search_feedback or {},
            ),
        )
        queries = self._extract_queries(payload)
        if len(queries) != self.max_queries_per_round:
            raise RuntimeError(
                "Local model returned JSON, but it did not contain a "
                f"`queries` list with exactly {self.max_queries_per_round} unique "
                "queries for query refinement."
            )
        return queries

    def analyze_search_feedback(
        self,
        original_query: str,
        refined_query: str,
        user_feedback: str,
        validated_papers: list[dict[str, object]],
        used_queries: list[str],
    ) -> SearchFeedbackAnalysis:
        """Ask the local model to structure feedback before refining queries."""

        payload = self._expect_dict(
            "search feedback analysis",
            self._generate_json(
                "search feedback analysis",
                build_feedback_analysis_prompt(
                    original_query=original_query,
                    refined_query=refined_query,
                    user_feedback=user_feedback,
                    validated_papers=validated_papers[:20],
                    used_queries=used_queries,
                ),
            ),
        )

        revised_topic = self._required_string(
            payload,
            "revised_topic",
            "search feedback analysis",
        )
        return SearchFeedbackAnalysis(
            revised_topic=revised_topic[:240],
            positive_constraints=self._required_string_list(
                payload,
                "positive_constraints",
                "search feedback analysis",
            ),
            negative_constraints=self._required_string_list(
                payload,
                "negative_constraints",
                "search feedback analysis",
            ),
            query_strategy=self._required_string(
                payload,
                "query_strategy",
                "search feedback analysis",
            )[:500],
            reason=self._required_string(
                payload,
                "reason",
                "search feedback analysis",
            )[:500],
        )

    def validate_papers(
        self,
        context: SearchContext,
        papers: list[Paper],
        search_feedback: dict[str, object] | None = None,
        search_filters: dict[str, object] | None = None,
        validation_batch_size: int = 2,
    ) -> PaperValidationResult:
        """Ask the local model to semantically validate candidate papers."""

        if validation_batch_size < 1:
            raise ValueError("validation_batch_size must be greater than or equal to 1.")

        all_validations: list[PaperValidation] = []
        summaries: list[str] = []

        for batch_index, papers_batch in enumerate(_chunked(papers, validation_batch_size), start=1):
            result = self._validate_papers_batch(
                context=context,
                papers=papers_batch,
                search_feedback=search_feedback or {},
                search_filters=search_filters or {},
                batch_index=batch_index,
            )

            all_validations.extend(result.validated_papers)

            if result.summary:
                summaries.append(result.summary)

        return PaperValidationResult(
            validated_papers=all_validations,
            summary=" | ".join(summaries) if summaries else "model did not provide a summary",
        )
    def _validate_papers_batch(
        self,
        context: SearchContext,
        papers: list[Paper],
        search_feedback: dict[str, object],
        search_filters: dict[str, object],
        batch_index: int,
    ) -> PaperValidationResult:
        """Validate one batch of papers."""

        payload = self._expect_dict(
            "paper validation",
            self._generate_json(
                "paper validation",
                build_validate_papers_prompt(
                    context=context,
                    papers=papers,
                    search_feedback=search_feedback,
                    search_filters=search_filters,
                ),
            ),
        )

        raw_items = payload.get("validated_papers")

        if not isinstance(raw_items, list):
            raise RuntimeError(
                "Local model returned JSON, but it did not contain the required "
                "`validated_papers` list for paper validation."
            )

        validations: list[PaperValidation] = []

        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue

            paper_id = self._safe_string(raw_item, "paper_id")

            if not paper_id:
                continue

            validations.append(
                PaperValidation(
                    paper_id=paper_id,
                    relevance=self._normalize_relevance(
                        self._safe_string(raw_item, "relevance")
                    ),
                    decision=self._normalize_decision(
                        self._safe_string(raw_item, "decision"),
                        self._safe_string(raw_item, "relevance"),
                    ),
                    relevance_reason=self._safe_string(raw_item, "relevance_reason")[:400],
                    mismatch_reason=self._safe_string(raw_item, "mismatch_reason")[:400],
                    useful_for=self._safe_string(raw_item, "useful_for")[:400],
                )
            )

        return PaperValidationResult(
            validated_papers=validations,
            summary=self._optional_string(payload, "summary")
            or f"model did not provide a summary for validation batch {batch_index}",
        )
    def should_continue(
        self,
        context: SearchContext,
        round_number: int,
        validated_papers: list[dict[str, object]],
        last_new_paper_count: int,
        last_new_useful_count: int,
    ) -> ContinueDecision:
        """Ask the local model whether another round is worth trying."""

        payload = self._generate_json(
            "continue decision",
            build_continue_decision_prompt(
                context=context,
                round_number=round_number,
                validated_papers=validated_papers[:10],
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

    def _forced_query_context_assessment(
        self,
        user_query: str,
    ) -> QueryContextAssessment | None:
        """Correct common context-assessment mistakes from small local models."""

        text = self._normalized_query_text(user_query)
        tokens = set(text.split())

        if not text:
            return None

        if text.startswith("automated shopping") and not tokens.intersection(
            {"agent", "agents", "llm", "vlm", "ecommerce", "checkout", "recommendation", "retail"}
        ):
            return QueryContextAssessment(
                has_enough_context=False,
                reason=(
                    "the term automated shopping is ambiguous and may refer to "
                    "shopping agents, e-commerce automation, checkout automation, "
                    "recommendation systems, or retail operations"
                ),
            )

        general_phrases = {
            "geral",
            "general",
            "overview",
            "visao geral",
            "quero entender",
            "panorama",
            "broad",
            "review",
            "survey",
            "state of the art",
            "trend",
            "trends",
        }
        if any(phrase in text for phrase in general_phrases):
            return QueryContextAssessment(
                has_enough_context=True,
                reason="the user explicitly indicated a broad general direction for the topic",
            )

        specific_focus_terms = {
            "using",
            "with",
            "via",
            "lstm",
            "transformer",
            "rag",
            "llm",
            "vlm",
            "sentiment",
            "dataset",
            "datasets",
            "benchmark",
            "benchmarks",
            "metric",
            "metrics",
            "comparison",
            "evaluation",
            "deployment",
            "feature",
            "features",
            "model",
            "models",
        }
        if tokens.intersection(specific_focus_terms):
            return QueryContextAssessment(
                has_enough_context=True,
                reason="the query includes a specific research focus enough to guide the search",
            )

        task_terms = {
            "classification",
            "detection",
            "forecasting",
            "prediction",
            "recognition",
        }
        modality_terms = {
            "acoustic",
            "audio",
            "image",
            "multimedia",
            "multimodal",
            "sound",
            "speech",
            "text",
            "video",
            "visual",
        }

        if tokens.intersection(modality_terms) and tokens.intersection(task_terms):
            return QueryContextAssessment(
                has_enough_context=False,
                reason=(
                    "the query specifies a task and modality, but not whether the "
                    "user wants a general direction or a specific research focus"
                ),
            )

        if len(tokens) <= 3 and tokens.intersection(task_terms):
            return QueryContextAssessment(
                has_enough_context=False,
                reason=(
                    "the query is broad and does not specify whether the user wants "
                    "a general direction or a specific research focus"
                ),
            )

        return None

    def _normalized_query_text(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        return " ".join(re.findall(r"[a-z0-9]+", ascii_text.lower()))

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

    def _normalize_filter_payload(
        self,
        payload: dict[str, object],
        fallback_primary_intent: str,
    ) -> dict[str, object]:
        """Normalize older filter schemas into the official filter schema."""

        normalized = dict(payload)

        def merged(keys: list[str], max_length: int = 120) -> list[str]:
            items: list[str] = []
            seen: set[str] = set()
            for key in keys:
                for item in self._optional_string_list(
                    normalized,
                    key,
                    max_length=max_length,
                ):
                    item_key = item.lower()
                    if item_key in seen:
                        continue
                    seen.add(item_key)
                    items.append(item)
            return items

        primary_intent = self._optional_string(normalized, "primary_intent")
        conservative_filters = merged(
            [
                "conservative_filters",
                "required_concepts",
                "required_modality",
                "must_have",
            ]
        )
        expansive_filters = merged(
            [
                "expansive_filters",
                "positive_signals",
                "soft_preferences",
                "nice_to_have",
            ]
        )
        negative_constraints = merged(
            [
                "negative_constraints",
                "negative_signals",
                "hard_exclusion_rules",
                "must_not_have",
            ],
            max_length=300,
        )
        not_inferred = merged(["not_inferred"])
        validation_priority = merged(["validation_priority", "priority"])

        if not primary_intent and conservative_filters:
            primary_intent = " ".join(fallback_primary_intent.split())
        if primary_intent and not conservative_filters:
            conservative_filters = [primary_intent]
        if not primary_intent or not conservative_filters:
            raise RuntimeError(
                "Local model returned JSON for semantic filter planning, but it did "
                "not contain enough filter information to validate papers."
            )

        return {
            "primary_intent": primary_intent,
            "conservative_filters": conservative_filters,
            "expansive_filters": expansive_filters,
            "negative_constraints": negative_constraints,
            "not_inferred": not_inferred,
            "validation_priority": validation_priority,
            "reason": (
                self._optional_string(normalized, "reason")
                or "model did not provide a reason"
            ),
        }

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

    def _required_string_list(
        self,
        payload: dict[str, object],
        key: str,
        step_name: str,
        max_length: int = 120,
    ) -> list[str]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise RuntimeError(
                "Local model returned JSON, but it did not contain the required "
                f"`{key}` list for {step_name}."
            )

        strings: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = " ".join(str(item).split())
            key_text = text.lower()
            if not text or key_text in seen:
                continue
            seen.add(key_text)
            strings.append(text[:max_length])
        return strings

    def _optional_string_list(
        self,
        payload: dict[str, object],
        key: str,
        max_length: int = 120,
    ) -> list[str]:
        value = payload.get(key)
        if value is None:
            return []
        raw_items = value if isinstance(value, list) else [value]

        strings: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            text = " ".join(str(item).split())
            key_text = text.lower()
            if not text or key_text in seen:
                continue
            seen.add(key_text)
            strings.append(text[:max_length])
        return strings

    def _normalize_relevance(self, value: str) -> str:
        text = value.strip().lower()
        if text in {"high", "medium", "low", "reject"}:
            return text
        return "reject"

    def _normalize_decision(self, decision: str, relevance: str) -> str:
        text = decision.strip().lower()
        if text in {"include", "exclude"}:
            return text
        return "exclude" if self._normalize_relevance(relevance) == "reject" else "include"

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
