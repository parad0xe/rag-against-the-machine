from typing import Protocol

from src.domain.models.base import (
    Chunk,
    Document,
    File,
    ManifestFileCache,
)


class DocumentLoaderPort(Protocol):
    def load(
        self,
        file: File,
        chunk_size: int,
        cached_file: ManifestFileCache | None = None,
    ) -> Document: ...


class ChunksLoaderPort(Protocol):
    def load(self, chunk_ids: list[str]) -> dict[str, Chunk]: ...
