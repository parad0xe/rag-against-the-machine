import json
import logging
from pathlib import Path

from src.domain.models.base import Chunk, Document, ManifestFileCache
from src.utils.file import file_load_content, file_write_json

logger = logging.getLogger(__file__)


class RawIndexStoreSync:
    @property
    def name(self) -> str:
        return "Raw"

    @property
    def enable(self) -> bool:
        return self._enable

    @property
    def addition_enable(self) -> bool:
        return self._addition_enable

    def __init__(
        self,
        file_path: Path,
        enable: bool = True,
        addition_enable: bool = True,
    ) -> None:
        self._file_path = file_path
        self._dir_path = file_path.parent
        self._enable = enable
        self._addition_enable = addition_enable
        self._add_documents: list[Document] = []
        self._delete_chunk_ids: set[str] = set()

    def delete(self, expired_chunk_ids: set[str]) -> None:
        self._delete_chunk_ids.update(expired_chunk_ids)

    def track(
        self,
        document: Document,
        cached_file: ManifestFileCache | None = None,
    ) -> None:
        doc_changed = (
            not cached_file or cached_file.file_hash != document.file.hash
        )
        in_store = cached_file and self.name in cached_file.stores

        if doc_changed or not in_store:
            if in_store and cached_file:
                self._delete_chunk_ids.update(cached_file.chunk_ids)
            if self._addition_enable:
                self._add_documents.append(document)

    def commit(self, require_reset: bool = False) -> None:
        data: dict[str, Chunk] = {}
        if self._file_path.exists() and not require_reset:
            content = file_load_content(self._file_path)
            if content:
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    pass

        if self._delete_chunk_ids:
            for chunk_id in self._delete_chunk_ids:
                data.pop(chunk_id, None)

        if self._add_documents:
            for doc in self._add_documents:
                for cid, metadata in doc.chunk_metadatas.items():
                    data[cid] = metadata

        self._dir_path.mkdir(parents=True, exist_ok=True)
        file_write_json(self._file_path, data)

        self._add_documents.clear()
        self._delete_chunk_ids.clear()
