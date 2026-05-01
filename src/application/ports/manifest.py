from pathlib import Path
from typing import Protocol

from src.application.ports.reader import ReaderPort
from src.domain.models.base import Document, File, Manifest, ManifestFileCache


class ManifestManagerPort(Protocol):
    @property
    def manifest(self) -> Manifest: ...

    @property
    def expired_chunk_ids(self) -> set[str]: ...

    @property
    def fingerprint_mismatch(self) -> bool: ...

    def get(self, file: File) -> ManifestFileCache | None: ...

    def track(self, document: Document) -> None: ...

    def commit(self) -> None: ...


class ManifestStoragePort(ReaderPort[Manifest], Protocol):
    def save(self, file_path: Path, manifest: Manifest) -> None: ...
