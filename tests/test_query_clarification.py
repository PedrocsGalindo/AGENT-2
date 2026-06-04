import unittest

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.graph.nodes import (
    commit_enriched_query,
    handle_query_confirmation_or_revision,
    interpret_query_confirmation,
    route_after_initial_assessment,
    rewrite_user_query_after_clarification,
)
from academic_explorer_mvp.services.query_planner import QueryPlanner


class FakeModel:
    def __init__(self, *payloads: object) -> None:
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def generate_json(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return self.payloads.pop(0)


class QueryClarificationTests(unittest.TestCase):
    def test_interpret_query_confirmation_accepts_positive_answers(self) -> None:
        self.assertEqual(interpret_query_confirmation("sim"), "accepted")
        self.assertEqual(interpret_query_confirmation("Pode seguir"), "accepted")

    def test_interpret_query_confirmation_treats_long_text_as_revision(self) -> None:
        self.assertEqual(
            interpret_query_confirmation("audio violence detection with synthetic datasets"),
            "revised",
        )

    def test_interpret_query_confirmation_marks_short_unknown_text_unclear(self) -> None:
        self.assertEqual(interpret_query_confirmation("nao"), "unclear")

    def test_route_after_initial_assessment_uses_can_search_flag(self) -> None:
        self.assertEqual(
            route_after_initial_assessment({"query_enrichment": {"stage": "ready_to_search"}}),
            "ready_to_search",
        )
        self.assertEqual(
            route_after_initial_assessment(
                {"query_enrichment": {"stage": "awaiting_clarification_answer"}}
            ),
            "needs_clarification",
        )

    def test_commit_enriched_query_updates_context_only_after_acceptance(self) -> None:
        context = SearchContext(
            user_query="violence detection",
            min_year=2020,
            max_rounds=1,
            limit=5,
        )

        state = commit_enriched_query(
            {
                "context": context,
                "query_enrichment": {
                    "stage": "awaiting_query_confirmation",
                    "proposed_query": "audio-based violence detection",
                },
            }
        )

        self.assertEqual(state["context"].user_query, "audio-based violence detection")
        self.assertEqual(
            state["query_enrichment"]["resolved_query"],
            "audio-based violence detection",
        )
        self.assertEqual(state["query_enrichment"]["stage"], "committed")

    def test_rewrite_node_stores_proposal_in_query_enrichment(self) -> None:
        model = FakeModel(
            {
                "proposed_query": "audio-based violence detection",
                "message": "Com base na sua resposta, a busca ficaria: ...",
                "can_search": True,
                "reason": "The modality is now clear.",
            }
        )
        planner = QueryPlanner(model)
        state = rewrite_user_query_after_clarification(
            {
                "context": SearchContext(
                    user_query="violence detection",
                    min_year=2020,
                    max_rounds=1,
                    limit=5,
                ),
                "query_enrichment": {
                    "stage": "awaiting_clarification_answer",
                    "original_query": "violence detection",
                    "question": "Which modality should the search focus on?",
                    "reason": "The modality changes the literature.",
                    "answer": "audio",
                },
            },
            planner,
        )

        self.assertEqual(state["query_enrichment"]["stage"], "awaiting_query_confirmation")
        self.assertEqual(
            state["query_enrichment"]["proposed_query"],
            "audio-based violence detection",
        )
        self.assertIn("audio-based violence detection", state["query_enrichment"]["message"])

    def test_revision_confirmation_rewrites_proposal_and_waits_again(self) -> None:
        model = FakeModel(
            {
                "proposed_query": (
                    "audio violence detection using deep learning and synthetic datasets"
                ),
                "reason": "The user revision adds method and data constraints.",
            }
        )
        planner = QueryPlanner(model)
        state = handle_query_confirmation_or_revision(
            {
                "context": SearchContext(
                    user_query="violence detection",
                    min_year=2020,
                    max_rounds=1,
                    limit=5,
                ),
                "query_enrichment": {
                    "stage": "awaiting_query_confirmation",
                    "original_query": "violence detection",
                    "question": "Which modality should the search focus on?",
                    "reason": "The modality changes the literature.",
                    "answer": "audio",
                    "proposed_query": "audio-based violence detection",
                    "confirmation_answer": (
                        "quero violencia em audio usando deep learning e datasets sinteticos"
                    ),
                },
            },
            planner,
        )

        self.assertEqual(state["query_enrichment"]["stage"], "awaiting_query_confirmation")
        self.assertEqual(state["query_enrichment"]["confirmation_status"], "revised")
        self.assertEqual(
            state["query_enrichment"]["proposed_query"],
            "audio violence detection using deep learning and synthetic datasets",
        )

    def test_planner_assesses_vague_initial_query(self) -> None:
        model = FakeModel(
            {
                "can_search": False,
                "question": "Which modality should the search focus on?",
                "reason": "The modality changes the literature.",
            }
        )
        planner = QueryPlanner(model)

        assessment = planner.assess_initial_query(
            SearchContext(
                user_query="violence detection",
                min_year=2020,
                max_rounds=1,
                limit=5,
            )
        )

        self.assertFalse(assessment.can_search)
        self.assertEqual(assessment.question, "Which modality should the search focus on?")
        self.assertIn("violence detection", model.prompts[0])

    def test_planner_rewrites_user_query_after_clarification(self) -> None:
        model = FakeModel(
            {
                "proposed_query": "audio-based violence detection",
                "message": "Com base na sua resposta, a busca ficaria: ...",
                "can_search": True,
                "reason": "The modality is now clear.",
            }
        )
        planner = QueryPlanner(model)

        rewrite = planner.rewrite_user_query(
            initial_query="violence detection",
            question="Which modality should the search focus on?",
            question_reason="The modality changes the literature.",
            user_answer="audio",
        )

        self.assertEqual(rewrite.proposed_query, "audio-based violence detection")
        self.assertTrue(rewrite.can_search)
        self.assertIn("audio", model.prompts[0])


if __name__ == "__main__":
    unittest.main()
