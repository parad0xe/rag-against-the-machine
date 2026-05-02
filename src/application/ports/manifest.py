from pathlib import Path
from typing import Protocol

from src.application.ports.reader import ReaderPort
from src.domain.models.base import Document, File, Manifest, ManifestFileCache


class ManifestManagerPort(Protocol):
    """
    Port for managing the system's manifest state and document tracking.
    """

    @property
    def manifest(self) -> Manifest:
        """The current manifest model."""
        ...

    @property
    def expired_chunk_ids(self) -> set[str]:
        """Set of chunk IDs that are no longer referenced by the manifest."""
        ...

    @property
    def fingerprint_mismatch(self) -> bool:
        """Whether the current config differs from the loaded manifest."""
        ...

    def get(self, file: File) -> ManifestFileCache | None:
        """
        Retrieve cached information for a specific file.

        Args:
            file: The file to look up in the manifest.

        Returns:
            The cached file information or None if not found.
        """
        ...

    def track(self, document: Document) -> None:
        """
        Record document indexing state in the manifest.

        Args:
            document: The processed document to track.
        """
        ...

    def commit(self) -> None:
        """Persist the manifest state to storage."""
        ...


class ManifestStoragePort(ReaderPort[Manifest], Protocol):
    """
    Port for reading and writing manifest files to disk.
    """

    def save(self, file_path: Path, manifest: Manifest) -> None:
        """
        Save the manifest to the specified path.

        Args:
            file_path: The destination path.
            manifest: The manifest model to save.
        """
        ...
