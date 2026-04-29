from abc import ABC, abstractmethod

from src.domain.models.base import Document, ManifestFileCache


class BaseIndexStoreQuery(ABC):
    @property
    def name(self) -> str:
        return self._name

    @property
    def enable(self) -> bool:
        return self._enable

    @property
    def weight(self) -> float:
        return self._weight

    def __init__(
        self,
        name: str,
        enable: bool = True,
        weight: float = 1.0,
    ) -> None:
        self._name = name
        self._enable = enable
        self._weight = weight

    @abstractmethod
    def search(self, query: str, k: int) -> list[str] | None: ...


class BaseIndexStoreSync(ABC):
    @property
    def name(self) -> str:
        return self._name

    @property
    def addition_enable(self) -> bool:
        return self._addition_enable

    def __init__(
        self,
        name: str,
        addition_enable: bool = True,
    ) -> None:
        self._name = name
        self._addition_enable = addition_enable
        self._add_documents: list[Document] = []
        self._delete_chunk_ids: set[str] = set()

    def delete(self, expired_chunk_ids: set[str]) -> None:
        self._delete_chunk_ids.update(expired_chunk_ids)

    def track(
        self,
        document: Document,
        cached_file: ManifestFileCache | None = None,
    ) -> None:
        in_store = cached_file and self.name in cached_file.stores
        doc_changed = (
            not cached_file
            or cached_file.file_hash != document.file.hash
            or not in_store
        )

        if doc_changed:
            if in_store and cached_file:
                self._delete_chunk_ids.update(cached_file.chunk_ids)
                document.stores.discard(self.name)
            if self._addition_enable:
                self._add_documents.append(document)
                document.stores.add(self.name)

    @abstractmethod
    def commit(self, require_reset: bool = False) -> None: ...
