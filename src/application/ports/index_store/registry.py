from abc import ABC
from typing import Generic, TypeVar

from src.application.ports.index_store.store import (
    IndexStoreInterface,
    IndexStoreQueryInterface,
    IndexStoreSyncInterface,
)

T = TypeVar("T", bound=IndexStoreInterface)


class IndexStoreRegistryInterface(ABC, Generic[T]):
    def __init__(self, *stores: T) -> None:
        self.stores: tuple[T, ...] = stores

    @property
    def active_stores(self) -> list[T]:
        return [store for store in self.stores if store.enable]

    def get_store(self, name: str) -> T | None:
        for store in self.stores:
            if store.name == name:
                return store
        return None


class IndexStoreQueryRegistry(
    IndexStoreRegistryInterface[IndexStoreQueryInterface]
):
    pass


class IndexStoreSyncRegistry(
    IndexStoreRegistryInterface[IndexStoreSyncInterface]
):
    pass
