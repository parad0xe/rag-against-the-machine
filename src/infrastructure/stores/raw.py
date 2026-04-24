import json
import logging
from pathlib import Path
from typing import Any


from src.domain.models.document import Document, DocumentStatus
from src.infrastructure.stores.base import BaseStore
from src.utils.path_util import readfile

logger = logging.getLogger(__name__)


class RawChunkStore(BaseStore):
    @property
    def name(self) -> str:
        return "Raw"

    def __init__(self, filepath: Path, enable: bool = True) -> None:
        super().__init__(filepath.parent, enable)
        self._filepath = filepath

    def add(self, document: Document, status: DocumentStatus) -> None:
        if status == DocumentStatus.NOTHING_TO_DO:
            return
        super().add(document, status)

    def exists(self) -> bool:
        return super().exists() and self._filepath.exists()

    def commit(self, require_reset_before: bool) -> None:
        if not self.enable:
            return

        if not self._add_documents and not self._delete_chunk_ids:
            return

        logger.info(
            f"[{self.__class__.__name__}] Synchronizing local chunks..."
        )

        data: dict[str, Any] = {}
        if self._filepath.exists() and not require_reset_before:
            content = readfile(self._filepath, ignore_unicode_error=True)
            if content is not None:
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[{self.__class__.__name__}] Invalid JSON in "
                        f"{self._filepath.name}, starting fresh."
                    )

        if self._delete_chunk_ids:
            for chunk_id in self._delete_chunk_ids:
                data.pop(chunk_id, None)

        if self._add_documents:
            for doc in self._add_documents:
                for chunk_id, metadata in doc.chunk_metadatas.items():
                    data[chunk_id] = metadata

        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"[{self.__class__.__name__}] Saved {len(data)} chunks to "
            f"'{self._filepath.name}'."
        )

        self._clear_state()
