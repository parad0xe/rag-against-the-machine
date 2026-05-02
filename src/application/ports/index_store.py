from typing import Generator, Protocol, Sequence, TypeVar

from src.domain.models.base import Document, ManifestFileCache

T_co = TypeVar("T_co", covariant=True)


class IndexStoreQueryPort(Protocol):
    """
    Port for querying an index store.
    """

    @property
    def name(self) -> str:
        """The name of the index store."""
        ...

    @property
    def weight(self) -> float:
        """The relative weight of this store in multi-store results."""
        ...

    @property
    def enable(self) -> bool:
        """Whether this store is currently enabled for queries."""
        ...

    def search(self, query: str, k: int) -> list[str] | None:
        """
        Search the store for relevant documents.

        Args:
            query: The search query text.
            k: The number of results to retrieve.

        Returns:
            A list of chunk IDs or None if the search fails.
        """
        ...


class IndexStoreSyncPort(Protocol):
    """
    Port for synchronizing an index store with the document repository.
    """

    @property
    def name(self) -> str:
        """The name of the index store."""
        ...

    @property
    def addition_enable(self) -> bool:
        """Whether this store allows adding new documents."""
        ...

    @property
    def added_documents_count(self) -> int:
        """The number of documents staged for addition."""
        ...

    @property
    def added_chunks_count(self) -> int:
        """The number of chunks staged for addition."""
        ...

    @property
    def deleted_chunks_count(self) -> int:
        """The number of chunks staged for deletion."""
        ...

    def track(
        self,
        document: Document,
        cached_file: ManifestFileCache | None = None,
    ) -> None:
        """
        Track a document for potential indexing or update.

        Args:
            document: The document model to track.
            cached_file: The existing manifest cache for this file.
        """
        ...

    def delete(self, expired_chunk_ids: set[str]) -> None:
        """
        Stage specific chunk IDs for deletion.

        Args:
            expired_chunk_ids: A set of chunk IDs to remove.
        """
        ...

    def commit(
        self, require_reset: bool = False
    ) -> Generator[tuple[int, int, str], None, None]:
        """
        Commit staged changes (additions and deletions) to the store.

        Args:
            require_reset: Whether to completely clear the store before
                committing.

        Yields:
            Tuples of (current_step, total_steps, description).
        """
        ...


class IndexStoreRegistryPort(Protocol[T_co]):
    """
    Port for managing a collection of index stores.
    """

    @property
    def stores(self) -> Sequence[T_co]:
        """All registered stores."""
        ...

    @property
    def active_stores(self) -> Sequence[T_co]:
        """Stores that are currently enabled."""
        ...
