"""Shared validation and reporting helpers."""

from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.utils.text import clean_text

VALID_RELEVANCES = {"high", "medium", "low", "reject"}
RELEVANCE_ORDER = {"high": 0, "medium": 1, "low": 2, "reject": 3}
RELEVANCE_LABELS = {
    "high": "alta",
    "medium": "media",
    "low": "baixa",
    "reject": "rejeitada",
}


def normalize_relevance(value: object) -> str:
    """Normalize model relevance labels to the official vocabulary."""

    text = clean_text(value).lower()
    return text if text in VALID_RELEVANCES else "reject"


def normalize_decision(decision: object, relevance: object) -> str:
    """Normalize include/exclude decisions using relevance as fallback."""

    text = clean_text(decision).lower()
    if text in {"include", "exclude"}:
        return text
    return "exclude" if normalize_relevance(relevance) == "reject" else "include"


def force_reject_if_needed(
    relevance: str,
    decision: str,
    violates_negative_constraints: bool = False,
) -> tuple[str, str]:
    """Keep the rule that reject/negative-constraint papers must be excluded."""

    if violates_negative_constraints or relevance == "reject":
        return "reject", "exclude"
    return relevance, decision


def relevance_label(relevance: str) -> str:
    """Translate relevance labels for CLI output."""

    return RELEVANCE_LABELS.get(relevance, relevance or "desconhecida")


def validation_counts(
    validated: list[dict[str, object]],
    relevant: list[dict[str, object]],
    excluded: list[dict[str, object]],
    new_useful_count: int,
) -> dict[str, int]:
    """Count validation totals and relevance distribution."""

    counts = {
        "total": len(validated),
        "included": len(relevant),
        "excluded": len(excluded),
        "new_useful": new_useful_count,
        "high": 0,
        "medium": 0,
        "low": 0,
        "reject": 0,
    }
    for item in validated:
        relevance = clean_text(item.get("relevance")).lower()
        if relevance in VALID_RELEVANCES:
            counts[relevance] += 1
    return counts


def validation_counts_from_state(state: SearchState) -> dict[str, int]:
    """Read validation counts from state, recomputing them when needed."""

    counts = state.get("validation_counts")
    if isinstance(counts, dict) and counts:
        return {
            "total": int(counts.get("total") or 0),
            "included": int(counts.get("included") or 0),
            "excluded": int(counts.get("excluded") or 0),
            "new_useful": int(counts.get("new_useful") or 0),
            "high": int(counts.get("high") or 0),
            "medium": int(counts.get("medium") or 0),
            "low": int(counts.get("low") or 0),
            "reject": int(counts.get("reject") or 0),
        }

    validated = state.get("validated_papers", [])
    relevant = state.get("relevant_papers", [])
    excluded = state.get("excluded_papers", [])
    return validation_counts(
        validated=validated,
        relevant=relevant,
        excluded=excluded,
        new_useful_count=int(state.get("last_new_useful_count", 0) or 0),
    )


def validation_sort_key(item: dict[str, object]) -> tuple[int, int]:
    """Sort validation rows by relevance, then newest year."""

    paper = item.get("paper")
    year = getattr(paper, "year", None) or 0
    return (RELEVANCE_ORDER.get(clean_text(item.get("relevance")), 3), -year)
