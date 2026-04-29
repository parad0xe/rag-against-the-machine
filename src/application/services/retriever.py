import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

from src.application.ports.index_store import (
    IndexStoreQueryPort,
    IndexStoreRegistryPort,
)
from src.application.ports.llm import (
    LLMQueryExpanderPort,
    LLMReRankerPort,
    LLMTranslatorPort,
)
from src.application.ports.loader import ChunksLoaderPort
from src.domain.models.base import Chunk
from src.domain.models.dataset import MinimalSource, RagDataset
from src.domain.models.inference import MinimalSearchResults
from src.utils.common import md5

logger = logging.getLogger(__file__)


class RetrieverService:
    def __init__(
        self,
        index_store_registry: IndexStoreRegistryPort[IndexStoreQueryPort],
        chunks_loader: ChunksLoaderPort,
        translator: LLMTranslatorPort,
        reranker: LLMReRankerPort,
        expander: LLMQueryExpanderPort,
    ) -> None:
        self._index_store_registry = index_store_registry
        self._chunks_loader = chunks_loader
        self._translator = translator
        self._reranker = reranker
        self._expander = expander

    def retrieve_chunks(
        self,
        original_query: str,
        k: int = 10,
    ) -> list[Chunk]:
        # --- Query Translator
        translated_query = self._translator.translate_to_english(
            original_query
        )
        # --- Without Translator
        # translated_query = original_query
        # --------

        # --- With HyDE (Query expension)
        logger.info("Generating hypothetical document (HyDE)...")
        hypothetical_doc = self._expander.expand_query(translated_query)

        if hypothetical_doc:
            search_query = f"{translated_query}\n{hypothetical_doc}"
            logger.debug(f"Expanded Query:\n{search_query}")
        else:
            search_query = translated_query
        # --- Without HyDE
        # search_query = translated_query
        # ---------

        pool_size = max(k * 10, 50)
        search_results: list[tuple[list[str], float]] = []

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(store.search, search_query, k=pool_size): store
                for store in self._index_store_registry.active_stores
            }

            for future in futures:
                store = futures[future]
                res = future.result()
                if res:
                    search_results.append((res, store.weight))

        top_results = self.__compute_rrf(search_results)[:pool_size]
        top_ids = [cid for cid, _ in top_results]

        chunks_data = self._chunks_loader.load(top_ids)

        # --- With Reranker
        pool_chunks: list[Chunk] = []
        for chunk_id, _ in top_results:
            data: Chunk | None = chunks_data.get(chunk_id, None)
            if data is not None:
                pool_chunks.append(data)

        content_to_chunk: dict[str, Chunk] = {}
        chunk_texts: list[str] = []
        for chunk in pool_chunks:
            raw_content = chunk.get("text", "")
            file_path = chunk.get("file_path", "Unknown")

            if raw_content and raw_content not in content_to_chunk:
                content_to_chunk[raw_content] = chunk
                formatted_text_for_reranker = (
                    f"File: {file_path}\nCode:\n{raw_content}"
                )
                chunk_texts.append(formatted_text_for_reranker)

        best_texts = self._reranker.rerank(
            query=translated_query, chunks=chunk_texts, top_k=k
        )
        final_chunks = []
        for formatted_text in best_texts:
            original_content = formatted_text.split("Code:\n", 1)[1]
            final_chunks.append(content_to_chunk[original_content])
        return final_chunks
        # --- Without Reranker
        # pool_chunks: list[Chunk] = []
        # for chunk_id, _ in top_results[:k]:
        #    data: Chunk | None = chunks_data.get(chunk_id, None)
        #    if data is not None:
        #        pool_chunks.append(data)
        # return pool_chunks
        # ---------

    def search(
        self,
        original_query: str,
        k: int = 10,
        question_id: str | None = None,
    ) -> tuple[MinimalSearchResults, list[Chunk]]:
        chunks = self.retrieve_chunks(original_query=original_query, k=k)

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
        k: int = 10,
    ) -> Generator[tuple[MinimalSearchResults, list[Chunk]], None, None]:
        for question in dataset.rag_questions:
            minimal_search_results, chunks = self.search(
                original_query=question.question,
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
