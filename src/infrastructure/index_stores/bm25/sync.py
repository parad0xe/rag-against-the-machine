import logging
import shutil
from pathlib import Path

import bm25s

from src.domain.exceptions.base import RagError
from src.domain.models.base import Document, ManifestFileCache

logger = logging.getLogger(__file__)


class BM25IndexStoreSync:
    @property
    def name(self) -> str:
        return "BM25"

    @property
    def enable(self) -> bool:
        return self._enable

    @property
    def addition_enable(self) -> bool:
        return self._addition_enable

    def __init__(
        self,
        dir_path: Path,
        enable: bool = True,
        addition_enable: bool = True,
    ) -> None:
        self._dir_path = dir_path
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
        if cached_file and self.name in cached_file.stores:
            self._delete_chunk_ids.update(cached_file.chunk_ids)
        if self._addition_enable:
            self._add_documents.append(document)

    def commit(self, require_reset: bool = False) -> None:
        if not self._add_documents:
            raise RagError("No document to index for BM25")

        total_docs = len(self._add_documents)
        logger.info(
            f"[{self.__class__.__name__}] Building index for "
            f"{total_docs} documents..."
        )

        chunks: list[str] = []
        chunk_ids: list[dict[str, str]] = []

        for doc in self._add_documents:
            chunks.extend(doc.chunks)
            chunk_ids.extend({"id": cid} for cid in doc.chunk_ids)

        if self._dir_path.exists():
            shutil.rmtree(self._dir_path)
            self._dir_path.mkdir(parents=True, exist_ok=True)

        chunk_tokens = bm25s.tokenize(chunks)
        retriever = bm25s.BM25(corpus=chunk_ids)
        retriever.index(chunk_tokens)
        retriever.save(self._dir_path)

        logger.info(
            f"[{self.__class__.__name__}] Successfully saved index to "
            f"'{self._dir_path}'."
        )

        self._add_documents.clear()
        self._delete_chunk_ids.clear()
