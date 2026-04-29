from typing import Generic, Sequence, TypeVar

T = TypeVar("T")


class IndexStoreRegistry(Generic[T]):
    def __init__(self, *stores: T) -> None:
        self._stores = list(stores)

    @property
    def stores(self) -> Sequence[T]:
        return self._stores

    @property
    def active_stores(self) -> Sequence[T]:
        return [
            store for store in self._stores if getattr(store, "enable", False)
        ]
