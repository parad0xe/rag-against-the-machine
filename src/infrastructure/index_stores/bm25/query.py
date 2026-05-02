import logging
from pathlib import Path

import bm25s

from src.infrastructure.index_stores.base import BaseIndexStoreQuery

logger = logging.getLogger(__file__)


class BM25IndexStoreQuery(BaseIndexStoreQuery):
    """
    Search implementation using the BM25s library for lexical retrieval.
    """

    def __init__(
        self,
        dir_path: Path,
        enable: bool = True,
        weight: float = 1.0,
    ) -> None:
        """
        Initializes the BM25 query store.

        Args:
            dir_path: Path to the directory containing the BM25 index.
            enable: Whether this store is enabled for searching.
            weight: Relative weight for this store's results.
        """
        super().__init__(name="BM25", enable=enable, weight=weight)
        self._dir_path = dir_path
        self._retriever: bm25s.BM25 | None = None

    def search(self, query: str, k: int) -> list[str] | None:
        """
        Executes a lexical search using BM25.

        Args:
            query: The search query.
            k: The number of results to retrieve.

        Returns:
            A list of matching chunk IDs.
        """
        if self._retriever is None:
            self._retriever = bm25s.BM25.load(self._dir_path, load_corpus=True)

        corpus_size = (
            len(self._retriever.corpus) if self._retriever.corpus else 0
        )
        actual_k = min(max(1, k), corpus_size)

        if actual_k == 0:
            return []

        query_tokens = bm25s.tokenize([query], show_progress=False)
        results, _ = self._retriever.retrieve(
            query_tokens,
            k=actual_k,
            n_threads=1,
            show_progress=False,
        )

        if results.shape[1] == 0:
            return []

        return [doc.get("id") for doc in results[0]]
