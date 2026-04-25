import json
import logging
from pathlib import Path

from src.infrastructure.document.stores.base import BaseSyncIndexStore
from src.utils.path_util import file_write_json, readfile

logger = logging.getLogger(__file__)


class RawSyncIndexStore(BaseSyncIndexStore):
    @property
    def name(self) -> str:
        return "Raw"

    def __init__(self, filepath: Path, enable: bool = True) -> None:
        super().__init__(filepath.parent, enable)
        self._filepath = filepath

    def commit(self, require_reset_before: bool) -> None:
        if not self.enable:
            return

        if not self._add_documents and not self._delete_chunk_ids:
            return

        data: dict = {}
        if self._filepath.exists() and not require_reset_before:
            content = readfile(self._filepath, ignore_unicode_error=True)
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

        file_write_json(self._filepath, data)
        self._clear_state()
