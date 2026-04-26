import json
import logging
from pathlib import Path

from src.application.ports.index_store.store import IndexStoreSyncInterface
from src.utils.file import file_load_content, file_write_json

logger = logging.getLogger(__file__)


class RawIndexStoreSync(IndexStoreSyncInterface):
    @property
    def name(self) -> str:
        return "Raw"

    def __init__(
        self,
        file_path: Path,
        enable: bool = True,
        addition_enable: bool = True,
    ) -> None:
        super().__init__(file_path.parent, enable, addition_enable)
        self._file_path = file_path

    def _perform_commit(self, require_reset_before: bool) -> None:
        data: dict = {}
        if self._file_path.exists() and not require_reset_before:
            content = file_load_content(
                self._file_path, ignore_unicode_error=True
            )
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

        file_write_json(self._file_path, data)
        self._clear_state()
