import logging

from src.infrastructure.indexers.stores.base import BaseIndexStore

logger = logging.getLogger(__file__)


class BM25IndexStore(BaseIndexStore):
    def delete(self, _: set[str]) -> None:
        pass

    def commit(self, _: bool) -> None:
        if not self._enable or not self._add_documents:
            return

        import bm25s

        total_docs = len(self._add_documents)
        logger.info(
            f"[{self.__class__.__name__}] Building index for "
            f"{total_docs} documents..."
        )

        chunks: list[str] = []
        chunk_ids: list[dict[str, list[str]]] = []
        for doc in self._add_documents:
            chunks.extend(doc.chunks)
            chunk_ids.append({"id": doc.chunk_ids})

        chunk_tokens = bm25s.tokenize(chunks)
        retriever = bm25s.BM25(corpus=chunk_ids)
        retriever.index(chunk_tokens)
        retriever.save(self._dirpath)

        logger.info(
            f"[{self.__class__.__name__}] Successfully saved index to "
            f"'{self._dirpath}'."
        )

        self._clear_state()
