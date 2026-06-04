from dataclasses import dataclass, field

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import RankedPaper


@dataclass(frozen=True)
class PromptSpec:
    """Versioned prompt used by the local model."""

    name: str
    version: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


INITIAL_QUERIES_PROMPT_VERSION = "1.0.1"
REFINE_QUERIES_PROMPT_VERSION = "1.0.0"
CONTINUE_DECISION_PROMPT_VERSION = "1.0.0"
ENRICH_QUERY_PROMPT_VERSION = "1.0.0"
REWRITE_USER_QUERY_PROMPT_VERSION = "1.0.0"
REWRITE_FROM_USER_REVISION_PROMPT_VERSION = "1.0.0"


def build_enrich_query_prompt(context: SearchContext) -> PromptSpec:
    """Prompt for assessing whether the user's topic is searchable."""

    text = f"""Return only one JSON object. No markdown. No explanation.
                Decide whether the user query has enough context to start a useful
                academic search.

                A query is sufficient when it contains at least one clear direction:
                domain, modality, application, method, dataset, restriction, or
                phenomenon.

                If the query is vague, ask exactly one short clarification question.
                Do not ask for minor details. If the query already has a domain,
                problem, method, or application, prefer can_search=true.

                User query: {context.user_query}

                If searchable, use this exact shape:
                {{"can_search":true,"question":null,"reason":"short reason"}}

                If clarification is needed, use this exact shape:
                {{"can_search":false,"question":"short clarification question","reason":"why this question matters"}}
            """

    return PromptSpec(
        name="enrich_query",
        version=ENRICH_QUERY_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "initial_query_assessment",
        },
    )


def build_rewrite_user_query_prompt(
    initial_query: str,
    question: str,
    question_reason: str,
    user_answer: str,
) -> PromptSpec:
    """Prompt for proposing one refined user topic after clarification."""

    text = f"""Return only one JSON object. No markdown. No explanation.
                Create one clearer academic search topic from the initial query and
                the user's clarification answer.

                This output is not a list of academic search queries. It is the
                refined user topic that will later be passed to initial query
                planning.

                Rules:
                - Create only one clearer academic topic.
                - Do not create multiple queries.
                - Do not invent information.
                - Use only the initial query and the user's answer.

                Initial query: {initial_query}
                Clarification question: {question}
                Why the question matters: {question_reason}
                User answer: {user_answer}

                Exact shape:
                {{"proposed_query":"clear academic search topic","message":"Com base na sua resposta, a busca ficaria: ...","can_search":true,"reason":"short reason"}}
            """

    return PromptSpec(
        name="rewrite_user_query",
        version=REWRITE_USER_QUERY_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "clarified_user_query_rewrite",
        },
    )


def build_rewrite_from_user_revision_prompt(
    initial_query: str,
    proposed_query: str,
    clarification_question: str,
    clarification_reason: str,
    clarification_answer: str,
    user_revision: str,
) -> PromptSpec:
    """Prompt for refining a user-written revision of the proposed topic."""

    text = f"""Return only one JSON object. No markdown. No explanation.
                Create one clear academic search topic from the user's revision.

                Rules:
                - Prefer the revision written by the user.
                - Preserve useful context from the initial query and clarification.
                - Do not invent methods, datasets, domains, or restrictions.
                - Do not create multiple queries.
                - This is still the refined user topic, not generated academic
                  search queries.

                Initial query: {initial_query}
                Previous proposed query: {proposed_query}
                Clarification question: {clarification_question}
                Clarification reason: {clarification_reason}
                Clarification answer: {clarification_answer}
                User revision: {user_revision}

                Exact shape:
                {{"proposed_query":"clear academic search topic","reason":"short reason"}}
            """

    return PromptSpec(
        name="rewrite_from_user_revision",
        version=REWRITE_FROM_USER_REVISION_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "user_revision_query_rewrite",
        },
    )

def build_initial_queries_prompt(context: SearchContext) -> PromptSpec:
    """Prompt for initial search queries."""

    text = f"""Return only one JSON object. No markdown. No explanation.
                The "queries" array must contain 1 to 3 non-empty strings.
                Use this topic in the query text: {context.user_query}
                Minimum year: {context.min_year}
                Exact shape: {{"queries":["{context.user_query}"],"reason":"core topic"}}
            """

    return PromptSpec(
        name="initial_queries",
        version=INITIAL_QUERIES_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "initial_query_planning",
        },
    )


def build_refine_queries_prompt(
    context: SearchContext,
    ranked_papers: list[RankedPaper],
    used_queries: list[str],
) -> PromptSpec:
    """Prompt for refining search queries."""

    titles = "; ".join(item.paper.title[:80] for item in ranked_papers) or "none"
    used = "; ".join(used_queries[-6:]) or "none"

    text = f"""Return only one JSON object. No markdown. No explanation.
                The "queries" array must contain 1 to 3 non-empty strings.
                Create new academic search queries. Do not repeat used queries.
                Use this topic in the query text: {context.user_query}
                Minimum year: {context.min_year}
                Used queries: {used}
                Best paper titles: {titles}
                Exact shape: {{"queries":["{context.user_query} method"],"reason":"new angle"}}
            """

    return PromptSpec(
        name="refine_queries",
        version=REFINE_QUERIES_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "query_refinement",
        },
    )


def build_continue_decision_prompt(
    context: SearchContext,
    round_number: int,
    ranked_papers: list[RankedPaper],
    last_new_paper_count: int,
    last_new_useful_count: int,
) -> PromptSpec:
    """Prompt for deciding whether another search round is useful."""

    titles = "; ".join(item.paper.title[:80] for item in ranked_papers) or "none"

    text = f"""Return only one JSON object. No markdown. No explanation.
                Decide if another academic search round is useful.
                Topic: {context.user_query}
                Round: {round_number} of {context.max_rounds}
                New papers: {last_new_paper_count}
                New useful papers: {last_new_useful_count}
                Best paper titles: {titles}
                The "continue" value must be true or false.
                Exact shape: {{"continue":true,"reason":"short reason"}}
            """

    return PromptSpec(
        name="continue_decision",
        version=CONTINUE_DECISION_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "continue_decision",
        },
    )
