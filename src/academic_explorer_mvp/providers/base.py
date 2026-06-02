"""Provider protocol for academic search APIs."""



from typing import Protocol

from academic_explorer_mvp.domain.paper import RawPaper


class SearchProvider(Protocol):
    """Small interface implemented by provider adapters."""

    name: str

    def search(self, query: str, min_year: int, limit: int) -> list[RawPaper]:
        """Search papers and return raw provider payloads."""

        ...
