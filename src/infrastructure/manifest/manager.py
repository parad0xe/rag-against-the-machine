from src.domain.models.document import Document, DocumentStatus
from src.domain.models.manifest import (
    Manifest,
    ManifestFileCache,
)
from src.infrastructure.manifest.storages.base import BaseManifestStorage


class ManifestManager:
    def __init__(
        self,
        storage: BaseManifestStorage,
    ) -> None:
        self._storage = storage

        manifest, mismatch = storage.load()

        self.identity_mismatch: bool = mismatch
        self.manifest: Manifest = manifest

    def get_cached_file(self, document: Document) -> ManifestFileCache | None:
        cached_ext = self.manifest.files_by_ext.get(document.ext, {})
        return cached_ext.get(document.id, None)

    def diff(self, document: Document) -> list[str] | None:
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
        self._storage.save(self.manifest)
