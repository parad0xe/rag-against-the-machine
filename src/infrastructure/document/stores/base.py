from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

from src.domain.models.document import Document, DocumentStatus


class BaseIndexStore(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    def __init__(self, dirpath: Path, enable: bool = True) -> None:
        self.enable: bool = enable
        self._dirpath: Path = dirpath

    def exists(self) -> bool:
        return self._dirpath.exists()


class BaseQueryIndexStore(BaseIndexStore, ABC):
    @abstractmethod
    def search(self, query: str, k: int) -> list[str] | None: ...

    def __init__(
        self,
        dirpath: Path,
        enable: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(dirpath, enable)
        self.weight: float = weight


class BaseSyncIndexStore(BaseIndexStore, ABC):
    @abstractmethod
    def commit(self, require_reset_before: bool) -> None: ...

    def __init__(self, dirpath: Path, enable: bool = True) -> None:
        super().__init__(dirpath, enable)
        self._delete_chunk_ids: set[str] = set()
        self._add_documents: list[Document] = []
        self._document_ids: set[str] = set()

    def add(self, document: Document, status: DocumentStatus) -> None:
        if not self.enable or document.id in self._document_ids:
            return

        if status == DocumentStatus.NOTHING_TO_DO:
            return

        self._document_ids.add(document.id)
        self._add_documents.append(document)

    def delete(self, chunk_ids: set[str]) -> None:
        self._delete_chunk_ids.update(chunk_ids)

    def _clear_state(self) -> None:
        self._delete_chunk_ids.clear()
        self._add_documents.clear()
        self._document_ids.clear()

    def _batches(
        self, batch_size: int
    ) -> Generator[tuple[list[str], list[str]], None, None]:
        current_chunks: list[str] = []
        current_ids: list[str] = []

        for doc in self._add_documents:
            for chunk, chunk_id in zip(doc.chunks, doc.chunk_ids):
                current_chunks.append(chunk)
                current_ids.append(chunk_id)

                if len(current_chunks) >= batch_size:
                    yield current_chunks, current_ids
                    current_chunks, current_ids = [], []

        if current_chunks:
            yield current_chunks, current_ids
