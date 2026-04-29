from typing import TYPE_CHECKING, Generic, Sequence, TypeVar

if TYPE_CHECKING:
    from src.application.ports.index_store import (
        IndexStoreQueryPort,
        IndexStoreSyncPort,
    )

T = TypeVar("T", "IndexStoreQueryPort", "IndexStoreSyncPort")


class IndexStoreRegistry(Generic[T]):
    def __init__(self, *stores: T) -> None:
        self._stores: list[T] = list(stores)

    @property
    def stores(self) -> Sequence[T]:
        return self._stores

    @property
    def active_stores(self) -> Sequence[T]:
        return [
            store for store in self._stores if getattr(store, "enable", True)
        ]
