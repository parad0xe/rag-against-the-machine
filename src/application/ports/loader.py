from abc import abstractmethod
from pathlib import Path
from typing import Protocol

from src.domain.models.base import (
    Chunk,
    Document,
    File,
    Manifest,
    ManifestFileCache,
)
from src.domain.models.dataset import RagDataset


class FileLoaderPort(Protocol):
    def load(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> File | None: ...


class RagDatasetLoaderPort(Protocol):
    def load(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> RagDataset | None: ...


class DocumentLoaderPort(Protocol):
    @abstractmethod
    def load(
        self,
        file: File,
        chunk_size: int,
        cached_file: ManifestFileCache | None = None,
    ) -> Document: ...


class ChunksLoaderPort(Protocol):
    def load(self, chunk_ids: list[str]) -> dict[str, Chunk]: ...


class ManifestLoaderPort(Protocol):
    def load(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> Manifest | None: ...

    def load_with_properties(
        self,
        file_path: Path,
        repositories: list[Path],
        embedding_model_name: str,
        chunk_size: int,
        with_semantic: bool,
        fingerprint_seed: list[str | int | bool] | None = None,
    ) -> tuple[Manifest, bool]: ...
