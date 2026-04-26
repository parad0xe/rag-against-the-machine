import logging

import bm25s

from src.application.ports.index_store.store import IndexStoreSyncInterface
from src.domain.models.document import Document
from src.domain.models.manifest import ManifestFileCache

logger = logging.getLogger(__file__)


class BM25IndexStoreSync(IndexStoreSyncInterface):
    @property
    def name(self) -> str:
        return "BM25"

    def _requires_deletion(
        self,
        document: Document,
        cached_file: ManifestFileCache | None,
        document_has_changed: bool,
    ) -> bool:
        return bool(cached_file and self.name in cached_file.stores)

    def _requires_addition(
        self,
        document: Document,
        cached_file: ManifestFileCache | None,
        document_has_changed: bool,
    ) -> bool:
        return True

    def _perform_commit(self, require_reset: bool) -> None:
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

        chunk_tokens = bm25s.tokenize(chunks)
        retriever = bm25s.BM25(corpus=chunk_ids)
        retriever.index(chunk_tokens)
        retriever.save(self._dir_path)

        logger.info(
            f"[{self.__class__.__name__}] Successfully saved index to "
            f"'{self._dir_path}'."
        )

        self._clear_state()
