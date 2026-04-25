from abc import ABC
from typing import Generic, TypeVar

from src.infrastructure.document.stores.base import (
    BaseIndexStore,
    BaseQueryIndexStore,
    BaseSyncIndexStore,
)

T = TypeVar("T", bound=BaseIndexStore)


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


class QueryIndexStoreRegistry(IndexStoreRegistry[BaseQueryIndexStore]):
    pass


class SyncIndexStoreRegistry(IndexStoreRegistry[BaseSyncIndexStore]):
    pass
