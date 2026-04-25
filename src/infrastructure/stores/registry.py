from src.infrastructure.stores.base import BaseStore


class StoreRegistry:
    def __init__(self, *stores: BaseStore) -> None:
        self.stores = stores

    @property
    def active_stores(self) -> list[BaseStore]:
        return [store for store in self.stores if store.enable]

    def get_store(self, name: str) -> BaseStore | None:
        for store in self.stores:
            if store.name == name:
                return store
        return None
