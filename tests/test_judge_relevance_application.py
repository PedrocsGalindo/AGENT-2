import unittest

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import Paper
from academic_explorer_mvp.graph.nodes.paper_feedback_nodes import _build_feedback_message
from academic_explorer_mvp.graph.nodes.search_nodes import judge_paper_validations
from academic_explorer_mvp.services.query_planner import (
    JudgedPaperValidation,
    PaperValidationJudgeResult,
    QueryPlanner,
)


def _paper() -> Paper:
    return Paper(
        id="audio-emotion",
        title="Supervised machine learning for audio emotion recognition",
        abstract="Supervised models recognize emotions from audio signals.",
        year=2024,
        authors=[],
        source="test",
        source_id="audio-emotion",
        url=None,
        doi=None,
        citation_count=0,
    )


def _original_validation(paper: Paper) -> dict[str, object]:
    return {
        "paper": paper,
        "paper_id": paper.id,
        "relevance": "reject",
        "decision": "exclude",
        "relevance_reason": "",
        "mismatch_reason": "The paper is not directly about violence detection.",
        "useful_for": "",
    }


class _JudgeJsonModel:
    def generate_json(self, _: str) -> dict[str, object]:
        return {
            "judged_validations": [
                {
                    "paper_id": "audio-emotion",
                    "validation_is_correct": False,
                    "reason_is_supported": False,
                    "passes_conservative_filters": False,
                    "violates_negative_constraints": False,
                    "corrected_relevance": "low",
                    "corrected_decision": "include",
                    "judge_reason": (
                        "The paper is not about violence detection, but it studies "
                        "supervised machine learning for audio emotion recognition, "
                        "which may provide background methods for audio-based recognition."
                    ),
                }
            ],
            "summary": "One background paper was included with low relevance.",
        }


class _JudgePlanner:
    def judge_paper_validations(self, **_: object) -> PaperValidationJudgeResult:
        return PaperValidationJudgeResult(
            judged_validations=[
                JudgedPaperValidation(
                    paper_id="audio-emotion",
                    validation_is_correct=False,
                    reason_is_supported=False,
                    passes_conservative_filters=False,
                    violates_negative_constraints=False,
                    corrected_relevance="low",
                    corrected_decision="include",
                    judge_reason=(
                        "The paper is not about violence detection, but it studies "
                        "supervised machine learning for audio emotion recognition, "
                        "which may provide background methods for audio-based recognition."
                    ),
                )
            ],
            summary="One background paper was included with low relevance.",
        )


class JudgeRelevanceApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = SearchContext(
            user_query="audio violence detection",
            min_year=2020,
            max_rounds=2,
            limit=10,
        )
        self.paper = _paper()

    def test_parser_preserves_low_include_when_conservative_filters_fail(self) -> None:
        result = QueryPlanner(_JudgeJsonModel()).judge_paper_validations(
            context=self.context,
            papers=[self.paper],
            validated_papers=[_original_validation(self.paper)],
            search_filters={
                "conservative_filters": [
                    "violence detection",
                    "audio-based detection",
                ]
            },
        )

        judgment = result.judged_validations[0]
        self.assertFalse(judgment.passes_conservative_filters)
        self.assertEqual(judgment.corrected_relevance, "low")
        self.assertEqual(judgment.corrected_decision, "include")

    def test_node_applies_low_include_reason_counts_and_feedback(self) -> None:
        state = {
            "context": self.context,
            "deduplicated_papers": [self.paper],
            "validated_papers": [_original_validation(self.paper)],
            "search_filters": {},
            "search_feedback": {},
            "last_new_paper_ids": [self.paper.id],
        }

        judged_state = judge_paper_validations(state, _JudgePlanner())
        validated = judged_state["validated_papers"][0]

        self.assertEqual(validated["relevance"], "low")
        self.assertEqual(validated["decision"], "include")
        self.assertEqual(validated["relevance_reason"], validated["judge_reason"])
        self.assertEqual(len(judged_state["relevant_papers"]), 1)
        self.assertEqual(len(judged_state["excluded_papers"]), 0)
        self.assertEqual(judged_state["validation_counts"]["low"], 1)
        self.assertEqual(judged_state["validation_counts"]["included"], 1)
        self.assertEqual(judged_state["validation_counts"]["excluded"], 0)

        message = _build_feedback_message(judged_state)
        self.assertIn(self.paper.title, message)
        self.assertIn("Relevancia: baixa", message)
        self.assertIn(validated["judge_reason"], message)
        self.assertNotIn("Exemplos rejeitados pela validacao:", message)

    def test_negative_constraint_still_forces_exclusion(self) -> None:
        judgment = JudgedPaperValidation(
            paper_id=self.paper.id,
            validation_is_correct=False,
            reason_is_supported=True,
            passes_conservative_filters=False,
            violates_negative_constraints=True,
            corrected_relevance="low",
            corrected_decision="include",
            judge_reason="The paper violates an explicit negative constraint.",
        )

        class NegativeConstraintPlanner:
            def judge_paper_validations(
                self,
                **_: object,
            ) -> PaperValidationJudgeResult:
                return PaperValidationJudgeResult(
                    judged_validations=[judgment],
                    summary="Excluded by negative constraint.",
                )

        state = {
            "context": self.context,
            "deduplicated_papers": [self.paper],
            "validated_papers": [_original_validation(self.paper)],
            "search_filters": {},
            "search_feedback": {},
            "last_new_paper_ids": [self.paper.id],
        }
        judged_state = judge_paper_validations(state, NegativeConstraintPlanner())

        self.assertEqual(judged_state["validated_papers"][0]["relevance"], "reject")
        self.assertEqual(judged_state["validated_papers"][0]["decision"], "exclude")
        self.assertEqual(len(judged_state["relevant_papers"]), 0)
        self.assertEqual(len(judged_state["excluded_papers"]), 1)


if __name__ == "__main__":
    unittest.main()
