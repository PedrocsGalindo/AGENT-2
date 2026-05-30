"""Prompts used by the MVP local model."""

from __future__ import annotations

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import RankedPaper


def build_initial_queries_prompt(context: SearchContext) -> str:
    """Prompt for initial search queries."""

    return f"""Return only one JSON object. No markdown. No explanation.
The "queries" array must contain 1 to 3 non-empty strings.
Use this topic in the query text: {context.user_query}
Minimum year: {context.min_year}
Exact shape: {{"queries":["{context.user_query}"],"reason":"core topic"}}
"""


def build_refine_queries_prompt(
    context: SearchContext,
    ranked_papers: list[RankedPaper],
    used_queries: list[str],
) -> str:
    """Prompt for refining search queries."""

    titles = "; ".join(item.paper.title[:80] for item in ranked_papers) or "none"
    used = "; ".join(used_queries[-6:]) or "none"
    return f"""Return only one JSON object. No markdown. No explanation.
The "queries" array must contain 1 to 3 non-empty strings.
Create new academic search queries. Do not repeat used queries.
Use this topic in the query text: {context.user_query}
Minimum year: {context.min_year}
Used queries: {used}
Best paper titles: {titles}
Exact shape: {{"queries":["{context.user_query} method"],"reason":"new angle"}}
"""


def build_continue_decision_prompt(
    context: SearchContext,
    round_number: int,
    ranked_papers: list[RankedPaper],
    last_new_paper_count: int,
    last_new_useful_count: int,
) -> str:
    """Prompt for deciding whether another search round is useful."""

    titles = "; ".join(item.paper.title[:80] for item in ranked_papers) or "none"
    return f"""Return only one JSON object. No markdown. No explanation.
Decide if another academic search round is useful.
Topic: {context.user_query}
Round: {round_number} of {context.max_rounds}
New papers: {last_new_paper_count}
New useful papers: {last_new_useful_count}
Best paper titles: {titles}
The "continue" value must be true or false.
Exact shape: {{"continue":true,"reason":"short reason"}}
"""
