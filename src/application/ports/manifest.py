from pathlib import Path
from typing import Literal, Protocol, overload

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


class ManifestRepositoryPort(Protocol):
    def save(self, file_path: Path, manifest: Manifest) -> None: ...

    @overload
    def load(
        self, file_path: Path, ignore_errors: Literal[False] = False
    ) -> Manifest: ...

    @overload
    def load(
        self, file_path: Path, ignore_errors: Literal[True]
    ) -> Manifest | None: ...

    @overload
    def load(
        self, file_path: Path, ignore_errors: bool
    ) -> Manifest | None: ...

    def load(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> Manifest | None: ...
