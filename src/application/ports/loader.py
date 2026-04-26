from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.models.chunk import Chunk
from src.domain.models.document import Document
from src.domain.models.file import File
from src.domain.models.manifest import Manifest, ManifestFileCache


class FileLoaderInterface(ABC):
    @abstractmethod
    def load(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> File | None: ...


class DocumentLoaderInterface(ABC):
    @abstractmethod
    def load(
        self,
        file: File,
        chunk_size: int,
        cached_file: ManifestFileCache | None = None,
    ) -> Document: ...


class ChunksLoaderInterface(ABC):
    @abstractmethod
    def load(self, chunk_ids: list[str]) -> dict[str, Chunk]: ...


class ManifestLoaderInterface(ABC):
    @abstractmethod
    def load(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> Manifest | None: ...

    @abstractmethod
    def load_with_properties(
        self,
        file_path: Path,
        repositories: list[Path],
        embedding_model_name: str,
        chunk_size: int,
        fingerprint_seed: list | None = None,
    ) -> tuple[Manifest, bool]: ...
