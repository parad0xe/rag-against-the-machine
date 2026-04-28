from abc import ABC, abstractmethod

from src.domain.models.base import Document, File, Manifest, ManifestFileCache


class ManifestManagerInterface(ABC):
    @property
    @abstractmethod
    def manifest(self) -> Manifest: ...

    @property
    @abstractmethod
    def expired_chunk_ids(self) -> set[str]: ...

    @property
    @abstractmethod
    def fingerprint_mismatch(self) -> bool: ...

    @abstractmethod
    def get(self, file: File) -> ManifestFileCache | None: ...

    @abstractmethod
    def track(self, document: Document) -> None: ...

    @abstractmethod
    def commit(self) -> None: ...
