import unittest

from academic_explorer_mvp.domain.paper import Paper
from academic_explorer_mvp.graph.nodes.paper_feedback_nodes import (
    REJECTED_PAPERS_PREVIEW_LIMIT,
    _build_feedback_message,
)


def _paper(index: int) -> Paper:
    return Paper(
        id=f"paper-{index}",
        title=f"Rejected paper {index}",
        abstract=None,
        year=2025,
        authors=[],
        source="test",
        source_id=f"paper-{index}",
        url=None,
        doi=None,
        citation_count=0,
    )


class PaperFeedbackMessageTests(unittest.TestCase):
    def test_rejected_preview_shows_limit_total_relevance_and_reason(self) -> None:
        total_rejected = REJECTED_PAPERS_PREVIEW_LIMIT + 2
        excluded_papers = []
        for index in range(total_rejected):
            paper = _paper(index)
            excluded_papers.append(
                {
                    "paper": paper,
                    "paper_id": paper.id,
                    "relevance": "reject",
                    "corrected_relevance": "reject",
                    "decision": "exclude",
                    "judge_reason": f"Judge reason {index}.",
                    "judge_corrected_to_reject": True,
                }
            )

        message = _build_feedback_message(
            {
                "relevant_papers": [],
                "excluded_papers": excluded_papers,
                "search_filters": {},
                "validation_counts": {
                    "total": total_rejected,
                    "included": 0,
                    "excluded": total_rejected,
                    "new_useful": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "reject": total_rejected,
                },
            }
        )

        self.assertIn(
            (
                f"Mostrando {REJECTED_PAPERS_PREVIEW_LIMIT} de "
                f"{total_rejected} rejeitados."
            ),
            message,
        )
        self.assertIn(
            f"Rejected paper {REJECTED_PAPERS_PREVIEW_LIMIT - 1}",
            message,
        )
        self.assertNotIn(
            f"Rejected paper {REJECTED_PAPERS_PREVIEW_LIMIT}",
            message,
        )
        self.assertIn("Relevancia corrigida: rejeitada", message)
        self.assertIn("Motivo: Judge reason 0.", message)


if __name__ == "__main__":
    unittest.main()
