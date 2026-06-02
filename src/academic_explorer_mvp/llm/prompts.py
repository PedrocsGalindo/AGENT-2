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


INITIAL_QUERIES_PROMPT_VERSION = "1.0.0"
REFINE_QUERIES_PROMPT_VERSION = "1.0.0"
CONTINUE_DECISION_PROMPT_VERSION = "1.0.0"


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