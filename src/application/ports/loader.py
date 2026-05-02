from typing import Protocol

from src.domain.models.base import (
    Chunk,
    Document,
    File,
    ManifestFileCache,
)


class DocumentLoaderPort(Protocol):
    """
    Port for transforming raw files into structured documents with chunks.
    """

    def load(
        self,
        file: File,
        chunk_size: int,
        cached_file: ManifestFileCache | None = None,
    ) -> Document:
        """
        Process a file and split it into chunks.

        Args:
            file: The raw file to process.
            chunk_size: The maximum size of each chunk.
            cached_file: Optional cache from the manifest to preserve state.

        Returns:
            A Document containing chunks and metadata.
        """
        ...


class ChunksLoaderPort(Protocol):
    """
    Port for retrieving specific chunks from storage by their ID.
    """

    def load(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """
        Load chunk data for the specified IDs.

        Args:
            chunk_ids: List of chunk identifiers to load.

        Returns:
            A dictionary mapping chunk IDs to Chunk metadata.
        """
        ...
