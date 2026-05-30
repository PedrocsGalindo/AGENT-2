import unittest

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import Paper
from academic_explorer_mvp.services.ranker import PaperRanker


class PaperRankerTests(unittest.TestCase):
    def test_ranker_prefers_title_and_abstract_matches(self) -> None:
        context = SearchContext(
            user_query="audio violence detection",
            min_year=2020,
            max_rounds=1,
            limit=5,
        )
        papers = [
            Paper(
                id="1",
                title="Audio violence detection in surveillance recordings",
                abstract="A method for violence detection using audio features.",
                year=2024,
                authors=[],
                source="test",
                source_id="1",
                url="https://example.test/1",
                doi="10.1/example",
                citation_count=20,
            ),
            Paper(
                id="2",
                title="A survey of unrelated social topics",
                abstract=None,
                year=2019,
                authors=[],
                source="test",
                source_id="2",
                url=None,
                doi=None,
                citation_count=100,
            ),
        ]

        ranked = PaperRanker().rank(papers, context)

        self.assertEqual(ranked[0].paper.id, "1")
        self.assertGreater(ranked[0].score, ranked[1].score)
        self.assertTrue(any("title" in reason for reason in ranked[0].reasons))


if __name__ == "__main__":
    unittest.main()
