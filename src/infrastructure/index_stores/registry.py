from typing import TYPE_CHECKING, Generic, Sequence, TypeVar

from src.application.ports.index_store import IndexStoreRegistryPort

if TYPE_CHECKING:
    from src.application.ports.index_store import (
        IndexStoreQueryPort,
        IndexStoreSyncPort,
    )

T = TypeVar("T", "IndexStoreQueryPort", "IndexStoreSyncPort")


class IndexStoreRegistry(IndexStoreRegistryPort[T], Generic[T]):
    """
    Registry that holds a collection of index stores.
    """

    def __init__(self, *stores: T) -> None:
        """
        Initializes the registry with a set of stores.

        Args:
            *stores: The index stores to register.
        """
        self._stores: list[T] = list(stores)

    @property
    def stores(self) -> Sequence[T]:
        """All registered stores."""
        return self._stores

    @property
    def active_stores(self) -> Sequence[T]:
        """Stores that are currently enabled for operations."""
        return [
            store for store in self._stores if getattr(store, "enable", True)
        ]
