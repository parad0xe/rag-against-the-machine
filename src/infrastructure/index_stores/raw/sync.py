import json
import logging
from pathlib import Path
from typing import Generator

from src.domain.models.base import Chunk
from src.infrastructure.index_stores.base import BaseIndexStoreSync
from src.utils.file import file_load_content, file_write_json

logger = logging.getLogger(__file__)


class RawIndexStoreSync(BaseIndexStoreSync):
    """
    Synchronization implementation that saves raw chunk metadata to JSON.
    """

    def __init__(
        self,
        file_path: Path,
        addition_enable: bool = True,
    ) -> None:
        """
        Initializes the raw sync store.

        Args:
            file_path: Path to the JSON file where chunks are stored.
            addition_enable: Whether adding documents is allowed.
        """
        super().__init__(name="Raw", addition_enable=addition_enable)
        self._file_path = file_path
        self._dir_path = file_path.parent

    def commit(
        self, require_reset: bool = False
    ) -> Generator[tuple[int, int, str], None, None]:
        """
        Persists additions and deletions to the raw JSON file.

        Args:
            require_reset: Whether to clear the file before starting.

        Yields:
            Tuple of (current_step, total_steps, description).
        """
        yield 0, 1, "Saving chunks data"

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

        yield 1, 1, "Chunks saved"

        self._add_documents.clear()
        self._delete_chunk_ids.clear()
