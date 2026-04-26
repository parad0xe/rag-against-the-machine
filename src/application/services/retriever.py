from src.application.ports.index_store.registry import IndexStoreQueryRegistry
from src.application.ports.loader import ChunksLoaderInterface
from src.domain.models.chunk import Chunk
from src.domain.models.source import MinimalSource


class Retriever:
    def __init__(
        self,
        index_store_registry: IndexStoreQueryRegistry,
        chunks_loader: ChunksLoaderInterface,
    ) -> None:
        self._index_store_registry = index_store_registry
        self._chunks_loader = chunks_loader

    def search(self, query: str, k: int = 10) -> list[MinimalSource]:
        pool_size = max(k * 30, 200)
        search_results: list[tuple[list[str], float]] = []

        for store in self._index_store_registry.active_stores:
            res = store.search(query, k=pool_size)
            if res:
                search_results.append((res, store.weight))

        top_results = self.__compute_rrf(search_results)[:k]
        top_ids = [cid for cid, _ in top_results]

        chunks_data = self._chunks_loader.load(top_ids)

        sources: list[MinimalSource] = []
        for cid, _ in top_results:
            data: Chunk | None = chunks_data.get(cid, None)

            if data is None:
                continue

            source = MinimalSource(
                file_path=data.get("file_path", "Unknown"),
                first_character_index=data.get("first_character_index", -1),
                last_character_index=data.get("last_character_index", -1),
            )
            sources.append(source)

        return sources

    def __compute_rrf(
        self,
        ranked_lists: list[tuple[list[str], float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}

        for doc_list, weight in ranked_lists:
            for rank, doc_id in enumerate(doc_list):
                score = weight * (1.0 / (k + rank + 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        sorted_results = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )

        return sorted_results
