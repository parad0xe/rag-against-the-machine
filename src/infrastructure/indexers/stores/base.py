from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.models.document import Document, DocumentStatus


class BaseIndexStore(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    def __init__(self, dirpath: Path, enable: bool = True) -> None:
        self.enable: bool = enable

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
        # if not self.enable:
        #    return
        self._delete_chunk_ids.update(chunk_ids)

    @abstractmethod
    def commit(self, require_reset_before: bool) -> None: ...

    def _clear_state(self) -> None:
        self._delete_chunk_ids.clear()
        self._add_documents.clear()
        self._document_ids.clear()
