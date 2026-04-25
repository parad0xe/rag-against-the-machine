import logging
from pathlib import Path

import bm25s

from src.infrastructure.document.stores.base import BaseQueryIndexStore

logger = logging.getLogger(__file__)


class BM25QueryIndexStore(BaseQueryIndexStore):
    @property
    def name(self) -> str:
        return "BM25"

    def __init__(
        self,
        dirpath: Path,
        enable: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(dirpath, enable, weight)
        self._retriever: bm25s.BM25 | None = None

    def search(self, query: str, k: int) -> list[str] | None:
        if not self.enable or not self._dirpath.exists():
            return []

        if self._retriever is None:
            self._retriever = bm25s.BM25.load(self._dirpath, load_corpus=True)

        corpus_size = (
            len(self._retriever.corpus) if self._retriever.corpus else 0
        )
        actual_k = min(max(1, k), corpus_size)

        if actual_k == 0:
            return []

        query_tokens = bm25s.tokenize(query)
        results, _ = self._retriever.retrieve(
            query_tokens,
            k=actual_k,
            n_threads=4,
        )

        if results.shape[1] == 0:
            return []

        return [doc.get("id") for doc in results[0]]
