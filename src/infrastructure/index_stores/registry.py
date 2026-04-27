from src.application.ports.index_store.registry import (
    IndexStoreRegistryInterface,
)
from src.application.ports.index_store.store import (
    IndexStoreQueryInterface,
    IndexStoreSyncInterface,
)


class IndexStoreQueryRegistry(
    IndexStoreRegistryInterface[IndexStoreQueryInterface]
):
    pass


class IndexStoreSyncRegistry(
    IndexStoreRegistryInterface[IndexStoreSyncInterface]
):
    pass
