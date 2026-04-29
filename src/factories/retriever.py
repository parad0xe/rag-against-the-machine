from pathlib import Path

from src.application.ports.index_store import IndexStoreQueryPort
from src.application.ports.llm import LLMAssistantPort
from src.application.services.retriever import RetrieverService
from src.config import settings
from src.infrastructure.chunks.loader import ChunksJSONFileLoader
from src.infrastructure.index_stores.bm25.query import BM25IndexStoreQuery
from src.infrastructure.index_stores.chroma.query import (
    ChromaIndexStoreQuery,
)
from src.infrastructure.index_stores.registry import (
    IndexStoreRegistry,
)
from src.infrastructure.llm.assistants.qwen import QwenAssistantLLM
from src.infrastructure.llm.rerankers.cross_encoder import CrossEncoderReRanker
from src.infrastructure.llm.translators.text_generation import (
    TextGenerationTranslatorLLM,
)
from src.infrastructure.manifest.storage import ManifestJSONStorage


class RetrieverFactory:
    @staticmethod
    def build(
        bm25_dir_path: Path,
        chroma_dir_path: Path,
        chunks_file_path: Path,
        manifest_file_path: Path,
        embedding_model_name: str,
    ) -> tuple[RetrieverService, LLMAssistantPort]:

        manifest = ManifestJSONStorage().read(manifest_file_path)

        translator = TextGenerationTranslatorLLM()
        llm = QwenAssistantLLM(model_name=settings.llm_model)
        reranker = CrossEncoderReRanker()

        query_stores: list[IndexStoreQueryPort] = [
            BM25IndexStoreQuery(bm25_dir_path, weight=0.6),
            ChromaIndexStoreQuery(
                chroma_dir_path,
                embedding_model_name,
                enable=manifest.with_semantic,
                weight=0.4,
            ),
        ]

        retriever = RetrieverService(
            index_store_registry=IndexStoreRegistry(*query_stores),
            chunks_loader=ChunksJSONFileLoader(chunks_file_path),
            reranker=reranker,
            expander=llm,
            translator=translator,
        )

        return retriever, llm
