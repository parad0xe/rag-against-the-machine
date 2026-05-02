from pathlib import Path

from src.application.ports.index_store import IndexStoreQueryPort
from src.application.ports.llm.llm import LLMAssistantPort
from src.application.services.llm.assistant import AssistantService
from src.application.services.llm.query_expander import QueryExpanderService
from src.application.services.llm.query_translation import (
    QueryTranslatorService,
)
from src.application.services.llm.reranker import RerankerService
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
from src.infrastructure.llm.engines.cross_encoder import (
    CrossEncoderEngine,
)
from src.infrastructure.llm.engines.huggingface_causal import (
    HuggingFaceCausalEngine,
)
from src.infrastructure.llm.engines.huggingface_translation import (
    HuggingFaceTranslationEngine,
)
from src.infrastructure.manifest.storage import ManifestJSONStorage


class RetrieverFactory:
    """
    Factory class to instantiate and wire retriever components.
    """

    @staticmethod
    def build(
        bm25_dir_path: Path,
        chroma_dir_path: Path,
        chunks_file_path: Path,
        manifest_file_path: Path,
        embedding_model_name: str,
        llm_model: str = settings.llm_model,
        translator_model: str = settings.translator_model,
        cross_encoder_model: str = settings.cross_encoder_model,
    ) -> tuple[RetrieverService, LLMAssistantPort]:
        """
        Builds the retriever service and assistant with the given config.

        Args:
            bm25_dir_path: Path to the BM25 index.
            chroma_dir_path: Path to the ChromaDB storage.
            chunks_file_path: Path to the chunks metadata file.
            manifest_file_path: Path to the system manifest.
            embedding_model_name: Name of the embedding model.
            llm_model: Identifier for the causal LLM.
            translator_model: Identifier for the translation model.
            cross_encoder_model: Identifier for the re-ranking model.

        Returns:
            A tuple of (RetrieverService, LLMAssistantPort).
        """

        manifest = ManifestJSONStorage().read(manifest_file_path)

        causal_engine = HuggingFaceCausalEngine(model_name=llm_model)
        translation_engine = HuggingFaceTranslationEngine(
            model_name=translator_model
        )
        if manifest.with_semantic:
            cross_encoder_engine = CrossEncoderEngine(
                model_name=cross_encoder_model
            )
            reranker = RerankerService(
                cross_encoder_engine=cross_encoder_engine
            )
            expander = QueryExpanderService(llm_engine=causal_engine)
        else:
            reranker = None
            expander = None

        assistant = AssistantService(llm_engine=causal_engine)
        translator = QueryTranslatorService(
            translation_engine=translation_engine
        )

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
            expander=expander,
            translator=translator,
            extended=manifest.with_semantic,
        )

        return retriever, assistant
