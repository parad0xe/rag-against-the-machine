from typing import Generator

from src.application.ports.index_store import (
    IndexStoreQueryPort,
    IndexStoreRegistryPort,
)
from src.application.ports.llm import LLMTranslatorPort
from src.application.ports.loader import ChunksLoaderPort
from src.domain.models.base import Chunk
from src.domain.models.dataset import MinimalSource, RagDataset
from src.domain.models.inference import MinimalSearchResults
from src.utils.common import md5


class Retriever:
    def __init__(
        self,
        index_store_registry: IndexStoreRegistryPort[IndexStoreQueryPort],
        chunks_loader: ChunksLoaderPort,
    ) -> None:
        self._index_store_registry = index_store_registry
        self._chunks_loader = chunks_loader

    def retrieve_chunks(
        self,
        original_query: str,
        translator: LLMTranslatorPort,
        k: int = 10,
    ) -> list[Chunk]:
        translated_query = translator.translate_to_english(original_query)

        # pool_size = max(k * 30, 200)
        search_results: list[tuple[list[str], float]] = []

        for store in self._index_store_registry.active_stores:
            res = store.search(translated_query, k=k)
            if res:
                search_results.append((res, store.weight))

        top_results = self.__compute_rrf(search_results)[:k]
        top_ids = [cid for cid, _ in top_results]

        chunks_data = self._chunks_loader.load(top_ids)

        chunks: list[Chunk] = []
        for chunk_id, _ in top_results:
            data: Chunk | None = chunks_data.get(chunk_id, None)

            if data is None:
                continue

            chunks.append(data)

        return chunks

    def search(
        self,
        original_query: str,
        translator: LLMTranslatorPort,
        k: int = 10,
        question_id: str | None = None,
    ) -> tuple[MinimalSearchResults, list[Chunk]]:
        chunks = self.retrieve_chunks(
            original_query=original_query, translator=translator, k=k
        )

        sources: list[MinimalSource] = []
        for chunk in chunks:
            source = MinimalSource(
                file_path=chunk.get("file_path", "Unknown"),
                first_character_index=chunk.get("first_character_index", -1),
                last_character_index=chunk.get("last_character_index", -1),
            )
            sources.append(source)

        question_id = (
            md5(original_query) if question_id is None else question_id
        )

        return MinimalSearchResults(
            question_id=question_id,
            question=original_query,
            retrieved_sources=sources,
        ), chunks

    def search_dataset_stream(
        self,
        dataset: RagDataset,
        translator: LLMTranslatorPort,
        k: int = 10,
    ) -> Generator[tuple[MinimalSearchResults, list[Chunk]], None, None]:
        for question in dataset.rag_questions:
            minimal_search_results, chunks = self.search(
                original_query=question.question,
                translator=translator,
                k=k,
                question_id=question.question_id,
            )
            yield minimal_search_results, chunks

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
