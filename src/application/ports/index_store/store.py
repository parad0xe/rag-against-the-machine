from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

from src.domain.models.document import Document
from src.domain.models.manifest import ManifestFileCache


class IndexStoreInterface(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    def __init__(self, dir_path: Path, enable: bool = True) -> None:
        self.enable: bool = enable
        self._dir_path: Path = dir_path

    def exists(self) -> bool:
        return self._dir_path.exists()


class IndexStoreQueryInterface(IndexStoreInterface, ABC):
    @abstractmethod
    def _perform_search(self, query: str, k: int) -> list[str] | None: ...

    def __init__(
        self,
        dir_path: Path,
        enable: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(dir_path, enable)
        self.weight: float = weight

    def search(self, query: str, k: int) -> list[str] | None:
        if not self.enable or not self._dir_path.exists():
            return []

        return self._perform_search(query, k)


class IndexStoreSyncInterface(IndexStoreInterface, ABC):
    @abstractmethod
    def _perform_commit(self, require_reset: bool) -> None: ...

    def __init__(
        self, dir_path: Path, enable: bool = True, addition_enable: bool = True
    ) -> None:
        super().__init__(dir_path, enable)
        self._addition_enable: bool = addition_enable
        self._delete_chunk_ids: set[str] = set()
        self._add_documents: list[Document] = []
        self._document_ids: set[str] = set()

    def commit(self, require_reset: bool) -> None:
        if not self.enable:
            return

        if not self._add_documents and not self._delete_chunk_ids:
            return

        return self._perform_commit(require_reset)

    def track(
        self,
        document: Document,
        cached_file: ManifestFileCache | None,
    ) -> None:
        if not self.enable:
            return

        document_has_changed = (
            not cached_file or cached_file.file_hash != document.file.hash
        )

        if self._requires_deletion(
            document, cached_file, document_has_changed
        ):
            if cached_file:
                self.delete(cached_file.chunk_ids)
            document.stores.discard(self.name)

        if self._requires_addition(
            document, cached_file, document_has_changed
        ):
            self.add(document)

    def add(self, document: Document) -> None:
        if not self.enable:
            return

        if not self._addition_enable or document.file.id in self._document_ids:
            return

        self._document_ids.add(document.file.id)
        self._add_documents.append(document)
        document.stores.add(self.name)

    def delete(self, chunk_ids: set[str]) -> None:
        if not self.enable:
            return
        self._delete_chunk_ids.update(chunk_ids)

    def _requires_deletion(
        self,
        document: Document,
        cached_file: ManifestFileCache | None,
        document_has_changed: bool,
    ) -> bool:
        return bool(cached_file and document_has_changed)

    def _requires_addition(
        self,
        document: Document,
        cached_file: ManifestFileCache | None,
        document_has_changed: bool,
    ) -> bool:
        return (
            not cached_file
            or document_has_changed
            or self.name not in cached_file.stores
            or not self.exists()
        )

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
