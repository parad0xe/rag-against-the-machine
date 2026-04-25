from src.domain.models.chunk import Chunk
from src.domain.models.source import MinimalSource
from src.infrastructure.document.stores.registry import QueryIndexStoreRegistry
from src.infrastructure.repositories.chunks import ChunksRepository
from src.utils.common_util import compute_rrf


class Retriever:
    def __init__(
        self,
        index_store_registry: QueryIndexStoreRegistry,
        chunks_repository: ChunksRepository,
    ) -> None:
        self._index_store_registry = index_store_registry
        self._chunks_repository = chunks_repository

    def search(self, query: str, k: int = 10) -> list[MinimalSource]:
        pool_size = max(k * 30, 200)
        search_results: list[tuple[list[str], float]] = []

        for store in self._index_store_registry.active_stores:
            res = store.search(query, k=pool_size)
            if res:
                search_results.append((res, store.weight))

        top_results = compute_rrf(search_results)[:k]
        top_ids = [cid for cid, _ in top_results]

        chunks_data = self._chunks_repository.get_chunks(top_ids)

        sources: list[MinimalSource] = []
        for cid, _ in top_results:
            data: Chunk | None = chunks_data.get(cid, None)

            if data is None:
                continue

            # On instancie le modèle strict pour chaque résultat
            source = MinimalSource(
                file_path=data.get("file_path", "Unknown"),
                first_character_index=data.get("first_character_index", -1),
                last_character_index=data.get("last_character_index", -1),
            )
            sources.append(source)

        return sources
