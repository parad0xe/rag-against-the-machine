from abc import abstractmethod
from pathlib import Path
from typing import Protocol

from src.domain.models.base import (
    Document,
    File,
    Manifest,
    ManifestFileCache,
)


class DocumentLoaderPort(Protocol):
    @abstractmethod
    def load(
        self,
        file: File,
        chunk_size: int,
        cached_file: ManifestFileCache | None = None,
    ) -> Document: ...


class ManifestLoaderPort(Protocol):
    def load(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> Manifest | None: ...
