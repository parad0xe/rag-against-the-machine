from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.models.document import Document, DocumentStatus


class BaseIndexStore(ABC):
    def __init__(self, dirpath: Path, enable: bool = True) -> None:
        self._dirpath: Path = dirpath
        self._delete_chunk_ids: set[str] = set()
        self._add_documents: list[Document] = []
        self._document_ids: set[str] = set()
        self._enable: bool = enable

    def add(self, document: Document, _: DocumentStatus) -> None:
        if not self._enable or document.id in self._document_ids:
            return
        self._document_ids.add(document.id)
        self._add_documents.append(document)

    def delete(self, chunk_ids: set[str]) -> None:
        if not self._enable:
            return
        self._delete_chunk_ids.update(chunk_ids)

    @abstractmethod
    def commit(self, require_reset_before: bool) -> None: ...
