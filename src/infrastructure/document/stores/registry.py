from abc import ABC
from typing import Generic, TypeVar

from src.infrastructure.document.stores.base import (
    IndexStore,
    IndexStoreQuery,
    IndexStoreSync,
)

T = TypeVar("T", bound=IndexStore)


class IndexStoreRegistry(ABC, Generic[T]):
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


class IndexStoreQueryRegistry(IndexStoreRegistry[IndexStoreQuery]):
    pass


class IndexStoreSyncRegistry(IndexStoreRegistry[IndexStoreSync]):
    pass
