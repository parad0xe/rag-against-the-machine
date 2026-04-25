import json
from pathlib import Path

from src.domain.exceptions.schema import (
    SchemaInvalidJSONFormatError,
    SchemaInvalidJSONRootError,
)
from src.domain.models.document import Document, DocumentStatus
from src.domain.models.manifest import Manifest, ManifestFileCache
from src.infrastructure.document.stores.base import IndexStoreSync
from src.infrastructure.document.stores.registry import IndexStoreSyncRegistry
from src.utils.common import compute_fingerprint
from src.utils.file import file_load_content, file_write_json


class ManifestRepository:
    def __init__(
        self,
        file_path: Path,
        repositories: list[Path],
        embedding_model_name: str,
        chunk_size: int,
        fingerprint_seed: list[str | int | bool] | None = None,
    ) -> None:
        self._file_path = file_path
        self._repositories = repositories
        self._embedding_model_name = embedding_model_name
        self._chunk_size = chunk_size
        self._fingerprint = compute_fingerprint(fingerprint_seed)

        manifest, mismatch = self._load()
        self.fingerprint_mismatch: bool = mismatch
        self.manifest: Manifest = manifest

    def _load(self) -> tuple[Manifest, bool]:
        default_data = {
            "repositories": [str(r) for r in self._repositories],
            "chunk_size": self._chunk_size,
            "embedding_model_name": self._embedding_model_name,
            "fingerprint": self._fingerprint,
        }

        if not self._file_path.exists():
            new_manifest = Manifest.model_validate(default_data)
            return new_manifest, False

        content = file_load_content(self._file_path)

        if not content:
            new_manifest = Manifest.model_validate(default_data)
            return new_manifest, False

        try:
            manifest_data = json.loads(content)

            if not isinstance(manifest_data, dict):
                raise SchemaInvalidJSONRootError(
                    expected=dict, context=self._file_path
                )
        except json.JSONDecodeError as e:
            raise SchemaInvalidJSONFormatError(
                context=self._file_path,
                lineno=e.lineno,
            ) from e

        manifest = Manifest.model_validate(manifest_data)
        fingerprint_mismatch = manifest.fingerprint != self._fingerprint

        merged_data = {**manifest_data, **default_data}
        merged_manifest = Manifest.model_validate(merged_data)

        return merged_manifest, fingerprint_mismatch

    def get_cached_file(self, document: Document) -> ManifestFileCache | None:
        cached_ext = self.manifest.files_by_ext.get(document.ext, {})
        return cached_ext.get(document.id, None)

    def diff(self, document: Document) -> list[str] | None:
        cached_file = self.get_cached_file(document)

        if not cached_file:
            return None

        if (
            not self.fingerprint_mismatch
            and cached_file.file_hash == document.hash
        ):
            return None

        return list(cached_file.chunk_ids)

    def update(
        self, document: Document, index_store_registry: IndexStoreSyncRegistry
    ) -> None:
        cached_file = self.get_cached_file(document)

        if not cached_file:
            self.manifest.files_by_ext.setdefault(document.ext, {})

        self.manifest.files_by_ext[document.ext][document.id] = (
            ManifestFileCache(
                file_path=document.file_path,
                file_hash=document.hash,
                chunk_ids=set(document.chunk_ids),
                stores={
                    store.name for store in index_store_registry.active_stores
                }.union(cached_file.stores if cached_file else set()),
            )
        )

    def get_status(
        self, document: Document, store: IndexStoreSync
    ) -> DocumentStatus:
        cached_file = self.get_cached_file(document)

        if (
            not cached_file
            or store.name not in cached_file.stores
            or not store.exists()
        ):
            return DocumentStatus.NEW

        if self.diff(document) is not None:
            return DocumentStatus.UPDATE

        return DocumentStatus.NOTHING_TO_DO

    def commit(self) -> None:
        file_write_json(
            self._file_path, self.manifest.model_dump_json(indent=2)
        )
