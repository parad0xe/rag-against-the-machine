import logging
from abc import ABC, abstractmethod
from typing import Generator

from src.application.ports.index_store import (
    IndexStoreQueryPort,
    IndexStoreSyncPort,
)
from src.domain.models.base import Document, ManifestFileCache

logger = logging.getLogger(__file__)


class BaseIndexStoreQuery(IndexStoreQueryPort, ABC):
    """
    Abstract base class for index store query implementations.
    """

    @property
    def name(self) -> str:
        """The name of the index store."""
        return self._name

    @property
    def enable(self) -> bool:
        """Whether this store is currently enabled for queries."""
        return self._enable

    @property
    def weight(self) -> float:
        """The relative weight of this store in multi-store results."""
        return self._weight

    def __init__(
        self,
        name: str,
        enable: bool = True,
        weight: float = 1.0,
    ) -> None:
        """
        Initializes the index store query implementation.

        Args:
            name: The name of the store.
            enable: Whether the store should be enabled for searching.
            weight: The relative importance of results from this store.
        """
        self._name = name
        self._enable = enable
        self._weight = weight

        if enable:
            logger.info(f"Use {self._name} for document retrieval.")

    @abstractmethod
    def search(self, query: str, k: int) -> list[str] | None:
        """
        Abstract method to search the index store.
        """
        ...


class BaseIndexStoreSync(IndexStoreSyncPort, ABC):
    """
    Abstract base class for index store synchronization implementations.
    """

    @property
    def name(self) -> str:
        """The name of the index store."""
        return self._name

    @property
    def addition_enable(self) -> bool:
        """Whether this store allows adding new documents."""
        return self._addition_enable

    @property
    def added_documents_count(self) -> int:
        """The number of documents staged for addition."""
        return len(self._add_documents)

    @property
    def added_chunks_count(self) -> int:
        """The number of chunks staged for addition."""
        return sum(len(d.chunks) for d in self._add_documents)

    @property
    def deleted_chunks_count(self) -> int:
        """The number of chunks staged for deletion."""
        return len(self._delete_chunk_ids)

    def __init__(
        self,
        name: str,
        addition_enable: bool = True,
    ) -> None:
        """
        Initializes the index store sync implementation.

        Args:
            name: The name of the store.
            addition_enable: Whether new documents can be added.
        """
        self._name = name
        self._addition_enable = addition_enable
        self._add_documents: list[Document] = []
        self._delete_chunk_ids: set[str] = set()

    def delete(self, expired_chunk_ids: set[str]) -> None:
        """
        Stages specific chunk IDs for deletion.

        Args:
            expired_chunk_ids: Set of IDs to remove.
        """
        self._delete_chunk_ids.update(expired_chunk_ids)

    def track(
        self,
        document: Document,
        cached_file: ManifestFileCache | None = None,
    ) -> None:
        """
        Identifies and stages changes for the provided document.

        Args:
            document: The document model.
            cached_file: Existing manifest cache for comparison.
        """
        in_store = cached_file and self.name in cached_file.stores
        doc_changed = (
            not cached_file
            or cached_file.file_hash != document.file.hash
            or not in_store
        )

        if doc_changed:
            if in_store and cached_file:
                self._delete_chunk_ids.update(cached_file.chunk_ids)
                document.stores.discard(self.name)
            if self._addition_enable:
                self._add_documents.append(document)
                document.stores.add(self.name)

    @abstractmethod
    def commit(
        self, require_reset: bool = False
    ) -> Generator[tuple[int, int, str], None, None]:
        """
        Abstract method to persist changes to the index store.
        """
        ...
