from abc import ABC, abstractmethod
from typing import Any

from src.domain.models.search import MinimalSearchResults


class SearchDisplayInterface(ABC):
    @abstractmethod
    def loader(self, message: str = "Searching...") -> Any: ...

    @abstractmethod
    def title(self, query: str) -> None: ...

    @abstractmethod
    def subquery(self, translated_query: str) -> None: ...

    @abstractmethod
    def noresult(self) -> None: ...

    @abstractmethod
    def results(self, search_result: MinimalSearchResults) -> None: ...
