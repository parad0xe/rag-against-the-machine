from pathlib import Path
from typing import Protocol

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


class ManifestStoragePort(Protocol):
    def save(self, file_path: Path, manifest: Manifest) -> None: ...

    def load(
        self,
        file_path: Path,
        embedding_model_name: str,
        repositories: list[Path],
        chunk_size: int,
        with_semantic: bool,
        fingerprint_seed: list[str | int | bool] | None = None,
    ) -> tuple[Manifest, bool]: ...
