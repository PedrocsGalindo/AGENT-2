"""Search planning and provider search nodes."""

from academic_explorer_mvp.domain.state import SearchState
from academic_explorer_mvp.graph.nodes.context_nodes import (
    _context,
    set_query_preview,
    set_search_filters,
)
from academic_explorer_mvp.services.query_planner import QueryPlanner
from academic_explorer_mvp.services.search_service import SearchService

from academic_explorer_mvp.services.deduplicator import PaperDeduplicator
from academic_explorer_mvp.services.normalizer import PaperNormalizer
from academic_explorer_mvp.utils.text import clean_text, find_constraint_hits
from academic_explorer_mvp.utils.validation import (
    force_reject_if_needed,
    validation_counts,
    validation_sort_key,
)


def plan_filters(state: SearchState, planner: QueryPlanner) -> SearchState:
    """Plan semantic validation filters from the current refined topic."""

    context = _context(state)
    filters = planner.plan_filters(context)
    new_state = set_search_filters(
        state,
        **filters.to_state(),
        stage="planned",
    )
    new_state["stop_reason"] = None
    return new_state


def plan_queries(state: SearchState, planner: QueryPlanner) -> SearchState:
    """Plan initial or refined queries with the local model."""

    context = _context(state)
    if state.get("round_number", 0) == 0:
        queries = planner.plan_initial_queries(context)
    else:
        queries = planner.refine_queries(
            context=context,
            validated_papers=state.get("validated_papers", []),
            used_queries=state.get("used_queries", []),
            search_feedback=state.get("search_feedback", {}) or {},
        )

    query_history = list(state.get("query_history", []))
    query_history.append(queries)

    new_state: SearchState = dict(state)
    new_state["pending_queries"] = queries
    new_state["query_history"] = query_history
    new_state["used_queries"] = [*state.get("used_queries", []), *queries]
    new_state = set_query_preview(
        new_state,
        stage="awaiting_query_preview",
        round=state.get("round_number", 0) + 1,
    )
    new_state["stop_reason"] = "awaiting query preview"
    return new_state


def search_papers(state: SearchState, search_service: SearchService) -> SearchState:
    """Search papers in OpenAlex and Semantic Scholar."""

    context = _context(state)
    queries = state.get("pending_queries", [])
    batch = search_service.search(queries=queries, min_year=context.min_year, limit=context.limit)

    new_state: SearchState = dict(state)
    new_state["round_number"] = state.get("round_number", 0) + 1
    new_state["raw_results"] = batch.raw_papers
    new_state["all_raw_results"] = [*state.get("all_raw_results", []), *batch.raw_papers]
    new_state["provider_errors"] = [*state.get("provider_errors", []), *batch.errors]
    new_state = set_query_preview(new_state, stage="idle")
    return new_state

def normalize_papers(state: SearchState, normalizer: PaperNormalizer) -> SearchState:
    """Normalize all cumulative raw papers."""

    normalized = normalizer.normalize_many(state.get("all_raw_results", []))
    new_state: SearchState = dict(state)
    new_state["normalized_papers"] = normalized
    return new_state


def deduplicate_papers(state: SearchState, deduplicator: PaperDeduplicator) -> SearchState:
    """Deduplicate normalized papers and count newly discovered papers."""

    previous_ids = set(state.get("known_paper_ids", []))
    deduplicated = deduplicator.deduplicate(state.get("normalized_papers", []))
    current_ids = {paper.id for paper in deduplicated}
    new_ids = current_ids - previous_ids

    new_state: SearchState = dict(state)
    new_state["deduplicated_papers"] = deduplicated
    new_state["known_paper_ids"] = sorted(current_ids)
    new_state["last_new_paper_count"] = len(new_ids)
    new_state["last_new_paper_ids"] = sorted(new_ids)
    return new_state


def validate_papers(
    state: SearchState,
    planner: QueryPlanner,
    validation_batch_size: int = 1,
) -> SearchState:
    """Validate papers semantically against the current search intent."""

    context = _context(state)
    papers = state.get("deduplicated_papers", [])

    result = planner.validate_papers(
        context=context,
        papers=papers,
        search_feedback=state.get("search_feedback", {}) or {},
        search_filters=state.get("search_filters", {}) or {},
        validation_batch_size=validation_batch_size,
    )

    validations_by_id = {item.paper_id: item for item in result.validated_papers}

    search_feedback = state.get("search_feedback", {}) or {}
    search_filters = state.get("search_filters", {}) or {}
    negative_constraints = [
        *_list_values(search_filters.get("negative_constraints")),
        *_list_values(search_feedback.get("negative_constraints")),
    ]

    validated: list[dict[str, object]] = []

    for paper in papers:
        validation = validations_by_id.get(paper.id)

        if validation is None:
            item = {
                "paper": paper,
                "paper_id": paper.id,
                "relevance": "reject",
                "decision": "exclude",
                "relevance_reason": "",
                "mismatch_reason": "The model did not return a validation for this paper.",
                "useful_for": "",
            }
        else:
            item = {
                "paper": paper,
                "paper_id": paper.id,
                "relevance": validation.relevance,
                "decision": validation.decision,
                "relevance_reason": validation.relevance_reason,
                "mismatch_reason": validation.mismatch_reason,
                "useful_for": validation.useful_for,
            }

        validated.append(_apply_negative_constraints(item, negative_constraints))

    validated = sorted(validated, key=validation_sort_key)

    relevant = [item for item in validated if item.get("decision") == "include"]
    excluded = [item for item in validated if item.get("decision") == "exclude"]

    new_ids = set(state.get("last_new_paper_ids", []))
    useful_relevances = {"high", "medium"}

    new_useful = [
        item
        for item in relevant
        if getattr(item.get("paper"), "id", None) in new_ids
        and item.get("relevance") in useful_relevances
    ]

    new_state: SearchState = dict(state)
    new_state["validated_papers"] = validated
    new_state["relevant_papers"] = relevant
    new_state["excluded_papers"] = excluded
    new_state["validation_counts"] = validation_counts(
        validated=validated,
        relevant=relevant,
        excluded=excluded,
        new_useful_count=len(new_useful),
    )
    new_state["validation_summary"] = (
        f"Validated {len(validated)} papers: "
        f"{len(relevant)} included and {len(excluded)} excluded."
    )
    new_state["model_validation_summary"] = result.summary
    new_state["ranked_papers"] = []
    new_state["last_new_useful_count"] = len(new_useful)

    return new_state


def judge_paper_validations(
    state: SearchState,
    planner: QueryPlanner,
    judge_batch_size: int = 1,
) -> SearchState:
    """Audit paper validations and apply the judge's corrected decisions."""

    print("[judge-validations] Auditing paper validations...")

    context = _context(state)
    papers = state.get("deduplicated_papers", [])
    original_validated = state.get("validated_papers", [])
    result = planner.judge_paper_validations(
        context=context,
        papers=papers,
        validated_papers=original_validated,
        search_filters=state.get("search_filters", {}) or {},
        search_feedback=state.get("search_feedback", {}) or {},
        judge_batch_size=judge_batch_size,
    )

    originals_by_id = {
        str(item.get("paper_id") or ""): item
        for item in original_validated
        if isinstance(item, dict)
    }
    judgments_by_id = {
        item.paper_id: item
        for item in result.judged_validations
    }
    judged: list[dict[str, object]] = []
    corrections = 0
    correction_logs: list[tuple[str, str, str]] = []

    for paper in papers:
        original = originals_by_id.get(paper.id) or {
            "paper": paper,
            "paper_id": paper.id,
            "relevance": "reject",
            "decision": "exclude",
            "relevance_reason": "",
            "mismatch_reason": "The original validator did not return this paper.",
            "useful_for": "",
        }
        judgment = judgments_by_id.get(paper.id)
        corrected = _apply_validation_judgment(original, judgment)
        judged.append(corrected)

        was_corrected = bool(corrected.get("judge_correction_applied"))
        if was_corrected:
            corrections += 1

        decision_changed = (
            str(original.get("decision") or "") != str(corrected.get("decision") or "")
            or str(original.get("relevance") or "") != str(corrected.get("relevance") or "")
        )
        if decision_changed:
            action = (
                "Corrected to reject"
                if corrected.get("decision") == "exclude"
                else "Corrected to include"
            )
            correction_logs.append(
                (
                    action,
                    _short_title(paper.title),
                    str(corrected.get("judge_reason") or "No reason provided."),
                )
            )

    judged = sorted(judged, key=validation_sort_key)
    relevant = [item for item in judged if item.get("decision") == "include"]
    excluded = [item for item in judged if item.get("decision") == "exclude"]
    new_ids = set(state.get("last_new_paper_ids", []))
    new_useful = [
        item
        for item in relevant
        if getattr(item.get("paper"), "id", None) in new_ids
        and item.get("relevance") in {"high", "medium"}
    ]

    print(f"[judge-validations] Total checked: {len(judged)}")
    print(f"[judge-validations] Corrections applied: {corrections}")
    print(f"[judge-validations] Included after judge: {len(relevant)}")
    print(f"[judge-validations] Excluded after judge: {len(excluded)}")
    for action, title, reason in correction_logs:
        print(f"[judge-validations] {action}: {title}")
        print(f"[judge-validations] Reason: {reason}")

    new_state: SearchState = dict(state)
    new_state["validated_papers"] = judged
    new_state["relevant_papers"] = relevant
    new_state["excluded_papers"] = excluded
    new_state["validation_counts"] = validation_counts(
        validated=judged,
        relevant=relevant,
        excluded=excluded,
        new_useful_count=len(new_useful),
    )
    new_state["validation_summary"] = (
        f"Judge audited {len(judged)} papers: "
        f"{len(relevant)} included and {len(excluded)} excluded "
        f"after {corrections} correction(s)."
    )
    new_state["judge_validation_summary"] = result.summary
    new_state["judge_corrections_count"] = corrections
    new_state["last_new_useful_count"] = len(new_useful)
    return new_state


def _apply_validation_judgment(
    original: dict[str, object],
    judgment: object | None,
) -> dict[str, object]:
    new_item = dict(original)
    original_relevance = str(original.get("relevance") or "reject")
    original_decision = str(original.get("decision") or "exclude")
    new_item["original_relevance"] = original_relevance
    new_item["original_decision"] = original_decision
    new_item["original_relevance_reason"] = str(
        original.get("relevance_reason") or ""
    )

    if judgment is None:
        validation_is_correct = False
        reason_is_supported = False
        passes_conservative_filters = False
        violates_negative_constraints = False
        corrected_relevance = "reject"
        corrected_decision = "exclude"
        judge_reason = "The judge did not return an audit for this paper."
    else:
        validation_is_correct = bool(
            getattr(judgment, "validation_is_correct", False)
        )
        reason_is_supported = bool(getattr(judgment, "reason_is_supported", False))
        passes_conservative_filters = bool(
            getattr(judgment, "passes_conservative_filters", False)
        )
        violates_negative_constraints = bool(
            getattr(judgment, "violates_negative_constraints", True)
        )
        corrected_relevance = str(
            getattr(judgment, "corrected_relevance", "reject")
        )
        corrected_decision = str(
            getattr(judgment, "corrected_decision", "exclude")
        )
        judge_reason = str(
            getattr(judgment, "judge_reason", "")
            or "The judge did not provide a reason."
        )

    corrected_relevance, corrected_decision = force_reject_if_needed(
        corrected_relevance,
        corrected_decision,
        violates_negative_constraints=violates_negative_constraints,
    )

    correction_applied = (
        not validation_is_correct
        or not reason_is_supported
        or corrected_relevance != original_relevance
        or corrected_decision != original_decision
    )
    corrected_to_reject = (
        corrected_decision == "exclude"
        and (
            original_decision != corrected_decision
            or original_relevance != corrected_relevance
        )
    )

    new_item.update(
        {
            "validation_is_correct": validation_is_correct,
            "reason_is_supported": reason_is_supported,
            "passes_conservative_filters": passes_conservative_filters,
            "violates_negative_constraints": violates_negative_constraints,
            "corrected_relevance": corrected_relevance,
            "corrected_decision": corrected_decision,
            "judge_reason": judge_reason,
            "judge_correction_applied": correction_applied,
            "judge_corrected_to_reject": corrected_to_reject,
            "relevance": corrected_relevance,
            "decision": corrected_decision,
        }
    )

    if corrected_decision == "exclude":
        new_item["relevance_reason"] = ""
        new_item["mismatch_reason"] = judge_reason
        new_item["useful_for"] = ""
    elif correction_applied:
        new_item["relevance_reason"] = judge_reason
        new_item["mismatch_reason"] = ""

    return new_item


def _short_title(title: str, max_length: int = 90) -> str:
    clean = " ".join(str(title).split())
    if len(clean) <= max_length:
        return clean
    return clean[: max_length - 3].rstrip() + "..."


def _list_values(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _apply_negative_constraints(
    item: dict[str, object],
    negative_constraints: list[object],
) -> dict[str, object]:
    hits = find_constraint_hits(_paper_text(item.get("paper")), negative_constraints)
    if not hits:
        return item

    new_item = dict(item)
    new_item["relevance"] = "reject"
    new_item["decision"] = "exclude"
    reason = "Matches excluded terms: " + ", ".join(hits[:4])
    existing = str(new_item.get("mismatch_reason") or "").strip()
    new_item["mismatch_reason"] = f"{existing} {reason}".strip()
    return new_item




def _paper_text(paper: object) -> str:
    title = clean_text(getattr(paper, "title", ""))
    abstract = clean_text(getattr(paper, "abstract", ""))
    return f"{title}\n{abstract}"
