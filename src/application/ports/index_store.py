from typing import Protocol, Sequence, TypeVar

from src.domain.models.base import Document, ManifestFileCache

T_co = TypeVar("T_co", covariant=True)


class IndexStoreQueryPort(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def weight(self) -> float: ...

    @property
    def enable(self) -> bool: ...

    def search(self, query: str, k: int) -> list[str] | None: ...


class IndexStoreSyncPort(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def addition_enable(self) -> bool: ...

    def track(
        self,
        document: Document,
        cached_file: ManifestFileCache | None = None,
    ) -> None: ...

    def delete(self, expired_chunk_ids: set[str]) -> None: ...

    def commit(self, require_reset: bool = False) -> None: ...


class IndexStoreRegistryPort(Protocol[T_co]):
    @property
    def stores(self) -> Sequence[T_co]: ...

    @property
    def active_stores(self) -> Sequence[T_co]: ...
