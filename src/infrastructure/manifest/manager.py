from pathlib import Path

from src.domain.models.document import Document, DocumentStatus
from src.domain.models.manifest import (
    Manifest,
    ManifestFileCache,
    RawManifestDict,
)
from src.infrastructure.manifest.storage import ManifestStorage
from src.utils.common_util import generate_identity


class ManifestManager:
    """
    Manages the state and caching logic of the manifest.
    """

    def __init__(
        self,
        filepath: Path,
        repositories: list[str],
        embedding_model_name: str,
        chunk_size: int,
        identity: list[str | int | bool] | None = None,
    ) -> None:
        """
        Initializes the manifest manager.

        Args:
            filepath: Location of the manifest file.
            repositories: Paths of the source code repositories.
            embedding_model_name: Name of the embedding model.
            chunk_size: Size of the text chunks.
            identity: Elements to generate the unique identity.
        """
        self._storage = ManifestStorage(filepath)
        self._identity: str = generate_identity(identity)

        self.identity_mismatch: bool = False
        self.manifest: Manifest = self.__load_or_create(
            repositories,
            embedding_model_name,
            chunk_size,
        )

    def get_cached_file(self, document: Document) -> ManifestFileCache | None:
        """
        Retrieves the cached file information for a document.

        Args:
            document: Target document for the lookup.

        Returns:
            The cached file information or None.
        """
        cached_ext = self.manifest.files_by_ext.get(document.ext, {})
        return cached_ext.get(document.id, None)

    def diff(self, document: Document) -> list[str] | None:
        """
        Computes the chunk differences for a document.

        Args:
            document: Target document to compare.

        Returns:
            Chunk identifiers to update, or None if unchanged.
        """
        cached_file = self.get_cached_file(document)

        if not cached_file:
            return None

        if (
            not self.identity_mismatch
            and cached_file.file_hash == document.hash
        ):
            return None

        return list(cached_file.chunk_ids)

    def update(self, document: Document) -> None:
        """
        Updates the manifest cache with a new document state.

        Args:
            document: Target document to update.
        """
        cached_file = self.get_cached_file(document)

        if not cached_file:
            self.manifest.files_by_ext.setdefault(document.ext, {})

        self.manifest.files_by_ext[document.ext][document.id] = (
            ManifestFileCache(
                file_path=document.filepath,
                file_hash=document.hash,
                chunk_ids=set(document.chunk_ids),
            )
        )

    def get_status(self, document: Document) -> DocumentStatus:
        cached_file = self.get_cached_file(document)

        if not cached_file:
            return DocumentStatus.NEW

        if self.diff(document) is not None:
            return DocumentStatus.UPDATE

        return DocumentStatus.NOTHING_TO_DO

    def commit(self) -> None:
        """
        Saves the current manifest state to the storage.
        """
        self._storage.save(self.manifest)

    def __load_or_create(
        self,
        repositories: list[str],
        embedding_model_name: str,
        chunk_size: int,
    ) -> Manifest:
        """
        Loads the manifest from storage or creates a default one.

        Args:
            repositories: Paths of the source code repositories.
            embedding_model_name: Name of the embedding model.
            chunk_size: Size of the text chunks.

        Returns:
            The loaded or newly created manifest.
        """
        default_manifest_data: RawManifestDict = {
            "repositories": list(
                {str(Path(repo).resolve()) for repo in repositories}
            ),
            "chunk_size": chunk_size,
            "embedding_model_name": embedding_model_name,
            "identity": self._identity,
        }

        manifest_data = self._storage.load()

        if not manifest_data:
            return Manifest.model_validate(default_manifest_data)

        manifest = Manifest.model_validate(manifest_data)
        if manifest.identity != self._identity:
            self.identity_mismatch = True

        return Manifest.model_validate(
            {**manifest_data, **default_manifest_data}
        )
