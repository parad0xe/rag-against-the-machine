from src.infrastructure.indexers.stores.base import BaseIndexStore


class BM25IndexStore(BaseIndexStore):
    def delete(self, _: set[str]) -> None:
        pass

    def commit(self, _: bool) -> None:
        if not self._enable:
            return

        chunks: list[str] = []
        chunk_ids: list[dict[str, list[str]]] = []
        for doc in self._add_documents:
            chunks.extend(doc.chunks)
            chunk_ids.append({"id": doc.chunk_ids})

        import bm25s

        chunk_tokens = bm25s.tokenize(chunks)
        retriever = bm25s.BM25(corpus=chunk_ids)
        retriever.index(chunk_tokens)
        retriever.save(self._dirpath)
