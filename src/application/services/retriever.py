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
        logger.debug(f"Original query received: '{original_query}'")

        # --- Query Translator
        translated_query = self._translator.translate_to_english(
            original_query
        )
        # --- Without Translator
        # translated_query = original_query
        # --------

        # --- With HyDE (Query expension)
        logger.info("Generating query expansion/keywords")
        keywords = self._expander.expand_query(translated_query)
        search_query = (
            f"{translated_query}\n{keywords}" if keywords else translated_query
        )
        logger.debug(f"Keywords extracted: '{keywords}'")
        logger.debug(f"Final search query:\n{search_query}")
        # --- Without HyDE
        # search_query = translated_query
        # ---------

        pool_size = max(k * 10, 50)
        search_results: list[tuple[list[str], float]] = []

        logger.debug(f"Querying active stores with pool_size={pool_size}")

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
                    logger.debug(
                        f"Store '{store.name}' returned {len(res)} results."
                    )
                else:
                    logger.debug(f"Store '{store.name}' returned 0 results.")

        logger.info("Performing RFF")
        top_results = self.__compute_rrf(search_results)[:pool_size]
        top_ids = [cid for cid, _ in top_results]
        logger.debug(f"Computed RRF. Kept top {len(top_ids)} unique chunks.")

        chunks_map = self._chunks_loader.load(top_ids)

        # --- With Reranker
        pool_chunks = [chunks_map[cid] for cid in top_ids if cid in chunks_map]
        logger.debug(
            f"Successfully loaded {len(pool_chunks)} chunks from storage."
        )

        if not pool_chunks:
            logger.debug("No chunks found. Returning empty list.")
            return []

        formatted_texts = [
            f"File: {c['file_path']}\nCode:\n{c['text']}" for c in pool_chunks
        ]

        text_to_chunk = dict(zip(formatted_texts, pool_chunks))

        logger.info("Performing re-ranking")
        logger.debug(
            f"Sending {len(formatted_texts)} chunks to the reranker "
            f"(top_k={k})"
        )
        best_texts = self._reranker.rerank(
            query=translated_query,
            chunks=formatted_texts,
            top_k=k,
        )
        logger.debug(f"Reranker returned {len(best_texts)} top chunks.")

        return [text_to_chunk[t] for t in best_texts if t in text_to_chunk]
        # --- Without Reranker
        # return [chunks_map[cid] for cid in top_ids[:k] if cid in chunks_map]
        # ---------

    def search(
        self,
        original_query: str,
        k: int = 10,
        question_id: str | None = None,
    ) -> tuple[MinimalSearchResults, list[Chunk]]:
        logger.debug(f"Starting search process (k={k})")
        chunks = self.retrieve_chunks(original_query=original_query, k=k)

        sources: list[MinimalSource] = [
            MinimalSource.model_validate(
                {
                    "file_path": c.get("file_path", "Unknown"),
                    "first_character_index": c.get(
                        "first_character_index", -1
                    ),
                    "last_character_index": c.get("last_character_index", -1),
                }
            )
            for c in chunks
        ]

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
        logger.debug(
            f"Streaming search for {len(dataset.rag_questions)} questions."
        )
        for question in dataset.rag_questions:
            yield self.search(
                original_query=question.question,
                k=k,
                question_id=question.question_id,
            )

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

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
