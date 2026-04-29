import json
import logging
from pathlib import Path

from src.domain.models.base import Chunk
from src.infrastructure.index_stores.base import BaseIndexStoreSync
from src.utils.file import file_load_content, file_write_json

logger = logging.getLogger(__file__)


class RawIndexStoreSync(BaseIndexStoreSync):
    def __init__(
        self,
        file_path: Path,
        addition_enable: bool = True,
    ) -> None:
        super().__init__(name="Raw", addition_enable=addition_enable)
        self._file_path = file_path
        self._dir_path = file_path.parent

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
