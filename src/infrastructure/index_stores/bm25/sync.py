import logging
from pathlib import Path
from typing import Generator

import bm25s

from src.domain.exceptions.base import RagError
from src.domain.models.base import Document, ManifestFileCache
from src.infrastructure.index_stores.base import BaseIndexStoreSync
from src.utils.file import safe_rmtree

logger = logging.getLogger(__file__)


class BM25IndexStoreSync(BaseIndexStoreSync):
    """
    Synchronization implementation for the BM25 lexical index.
    """

    def __init__(
        self,
        dir_path: Path,
        addition_enable: bool = True,
    ) -> None:
        """
        Initializes the BM25 sync store.

        Args:
            dir_path: Path where the BM25 index will be saved.
            addition_enable: Whether adding documents is allowed.
        """
        super().__init__(name="BM25", addition_enable=addition_enable)
        self._dir_path = dir_path

    def track(
        self,
        document: Document,
        cached_file: ManifestFileCache | None = None,
    ) -> None:
        """
        BM25 implementation of document tracking (always adds).

        Args:
            document: The document model to track.
            cached_file: Existing manifest cache for this file.
        """
        if self._addition_enable:
            self._add_documents.append(document)
            document.stores.add(self.name)

    def commit(
        self, require_reset: bool = False
    ) -> Generator[tuple[int, int, str], None, None]:
        """
        Rebuilds the entire BM25 index from staged documents.

        Args:
            require_reset: Unused for BM25 (always rebuilds).

        Yields:
            Tuple of (current_step, total_steps, description).

        Raises:
            RagError: If there are no documents staged for indexing.
        """
        if not self._add_documents:
            raise RagError("No document to index for BM25")

        total_docs = len(self._add_documents)
        yield 0, 1, f"Building index for {total_docs} docs"

        chunks: list[str] = []
        chunk_ids: list[dict[str, str]] = []

        for doc in self._add_documents:
            chunks.extend(doc.chunks)
            chunk_ids.extend({"id": cid} for cid in doc.chunk_ids)

        if self._dir_path.exists():
            safe_rmtree(self._dir_path)
            self._dir_path.mkdir(parents=True, exist_ok=True)

        chunk_tokens = bm25s.tokenize(chunks, show_progress=False)
        retriever = bm25s.BM25(corpus=chunk_ids)
        retriever.index(chunk_tokens, show_progress=False)
        retriever.save(self._dir_path, show_progress=False)

        yield 1, 1, "Index saved"

        self._add_documents.clear()
        self._delete_chunk_ids.clear()
