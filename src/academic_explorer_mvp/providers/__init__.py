"""Academic search providers."""

from academic_explorer_mvp.providers.openalex import OpenAlexProvider
from academic_explorer_mvp.providers.semantic_scholar import SemanticScholarProvider

__all__ = ["OpenAlexProvider", "SemanticScholarProvider"]
