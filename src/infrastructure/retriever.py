from typing import Any

from src.domain.models.source import MinimalSource
from src.infrastructure.stores.registry import StoreRegistry
from src.utils.common_util import compute_rrf


class Retriever:
    def __init__(self, registry: StoreRegistry) -> None:
        self._registry = registry

    def search(self, query: str, k: int = 10) -> list[MinimalSource]:
        pool_size = max(k * 30, 200)
        search_results: list[tuple[list[str], float]] = []

        for store in self._registry.active_stores:
            res = store.search(query, k=pool_size)
            if res:
                search_results.append((res, store.weight))

        top_results = compute_rrf(search_results)[:k]
        top_ids = [cid for cid, _ in top_results]

        chunks_data: dict[str, Any] = {}

        if top_ids:
            for store in self._registry.active_stores:
                items = store.get_items(top_ids)
                if items:
                    chunks_data.update(items)

        sources: list[MinimalSource] = []
        for cid, score in top_results:
            data = chunks_data.get(cid, {})

            # On instancie le modèle strict pour chaque résultat
            source = MinimalSource(
                file_path=data.get("file_path", "Unknown"),
                first_character_index=data.get("first_character_index", -1),
                last_character_index=data.get("last_character_index", -1),
            )
            sources.append(source)

        return sources
