from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.models.document import Document, DocumentStatus


class BaseStore(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def commit(self, require_reset_before: bool) -> None: ...

    def __init__(
        self,
        dirpath: Path,
        enable: bool = True,
        weight: float = 1.0,
    ) -> None:
        self.enable: bool = enable
        self.weight: float = weight

        self._dirpath: Path = dirpath
        self._delete_chunk_ids: set[str] = set()
        self._add_documents: list[Document] = []
        self._document_ids: set[str] = set()

    def add(self, document: Document, _: DocumentStatus) -> None:
        if not self.enable or document.id in self._document_ids:
            return
        self._document_ids.add(document.id)
        self._add_documents.append(document)

    def delete(self, chunk_ids: set[str]) -> None:
        self._delete_chunk_ids.update(chunk_ids)

    def exists(self) -> bool:
        return self._dirpath.exists()

    def search(self, query: str, k: int) -> list[str] | None:
        return None

    def get_items(self, chunk_ids: list[str]) -> dict[str, dict] | None:
        return None

    def _clear_state(self) -> None:
        self._delete_chunk_ids.clear()
        self._add_documents.clear()
        self._document_ids.clear()
