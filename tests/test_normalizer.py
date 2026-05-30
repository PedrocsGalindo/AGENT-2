import unittest

from academic_explorer_mvp.domain.paper import RawPaper
from academic_explorer_mvp.services.normalizer import PaperNormalizer


class PaperNormalizerTests(unittest.TestCase):
    def test_normalizes_openalex_payload(self) -> None:
        raw = RawPaper(
            source="openalex",
            source_id="https://openalex.org/W1",
            payload={
                "id": "https://openalex.org/W1",
                "title": "Audio Violence Detection",
                "publication_year": 2023,
                "cited_by_count": 12,
                "doi": "https://doi.org/10.123/test",
                "primary_location": {"landing_page_url": "https://example.test/openalex"},
                "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                "abstract_inverted_index": {"audio": [0], "violence": [1], "detection": [2]},
            },
        )

        paper = PaperNormalizer().normalize(raw)

        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.title, "Audio Violence Detection")
        self.assertEqual(paper.abstract, "audio violence detection")
        self.assertEqual(paper.year, 2023)
        self.assertEqual(paper.authors, ["Ada Lovelace"])
        self.assertEqual(paper.citation_count, 12)
        self.assertEqual(paper.url, "https://example.test/openalex")

    def test_normalizes_semantic_scholar_payload(self) -> None:
        raw = RawPaper(
            source="semantic_scholar",
            source_id="S1",
            payload={
                "paperId": "S1",
                "title": "Audio Event Detection",
                "abstract": "Detecting violent audio events.",
                "year": 2022,
                "citationCount": 7,
                "url": "https://example.test/semantic",
                "externalIds": {"DOI": "10.123/semantic"},
                "authors": [{"name": "Grace Hopper"}],
            },
        )

        paper = PaperNormalizer().normalize(raw)

        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.title, "Audio Event Detection")
        self.assertEqual(paper.abstract, "Detecting violent audio events.")
        self.assertEqual(paper.year, 2022)
        self.assertEqual(paper.authors, ["Grace Hopper"])
        self.assertEqual(paper.doi, "10.123/semantic")
        self.assertEqual(paper.citation_count, 7)


if __name__ == "__main__":
    unittest.main()
