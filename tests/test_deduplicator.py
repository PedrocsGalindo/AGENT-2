import unittest

from academic_explorer_mvp.domain.paper import Paper
from academic_explorer_mvp.services.deduplicator import PaperDeduplicator


def paper(
    title: str,
    *,
    doi: str | None = None,
    source: str = "test",
    source_id: str = "id",
    year: int | None = 2022,
) -> Paper:
    return Paper(
        id=f"{source}:{source_id}:{title}",
        title=title,
        abstract=None,
        year=year,
        authors=[],
        source=source,
        source_id=source_id,
        url=None,
        doi=doi,
        citation_count=0,
    )


class PaperDeduplicatorTests(unittest.TestCase):
    def test_deduplicates_by_doi(self) -> None:
        papers = [
            paper("First", doi="10.1/ABC", source_id="1"),
            paper("Second", doi="10.1/abc", source_id="2"),
        ]

        result = PaperDeduplicator().deduplicate(papers)

        self.assertEqual([item.title for item in result], ["First"])

    def test_deduplicates_by_source_and_source_id(self) -> None:
        papers = [
            paper("First", source="openalex", source_id="W1"),
            paper("Second", source="openalex", source_id="W1"),
        ]

        result = PaperDeduplicator().deduplicate(papers)

        self.assertEqual([item.title for item in result], ["First"])

    def test_deduplicates_by_title_and_year_when_ids_are_missing(self) -> None:
        papers = [
            paper("Audio Violence Detection", source_id="", year=2021),
            paper("Audio: violence detection!", source_id="", year=2021),
        ]

        result = PaperDeduplicator().deduplicate(papers)

        self.assertEqual([item.title for item in result], ["Audio Violence Detection"])


if __name__ == "__main__":
    unittest.main()
