from pathlib import Path

from src.application.ports.index_store import IndexStoreQueryPort
from src.application.services.retriever import Retriever
from src.domain.exceptions.storage import StorageFileNotFoundError
from src.infrastructure.chunks.json_loader import ChunksJSONLoader
from src.infrastructure.index_stores.bm25.query import BM25IndexStoreQuery
from src.infrastructure.index_stores.chroma.query import (
    ChromaIndexStoreQuery,
)
from src.infrastructure.index_stores.registry import (
    IndexStoreRegistry,
)
from src.infrastructure.llm.translators.text_generation import (
    TextGenerationTranslatorLLM,
)
from src.infrastructure.manifest.json_loader import ManifestJSONLoader


class RetrieverFactory:
    @staticmethod
    def build(
        bm25_dir_path: Path,
        chroma_dir_path: Path,
        chunks_file_path: Path,
        manifest_file_path: Path,
        embedding_model_name: str,
    ) -> tuple[Retriever, TextGenerationTranslatorLLM]:

        manifest = ManifestJSONLoader().load(manifest_file_path)
        if not manifest:
            raise StorageFileNotFoundError(manifest_file_path)

        query_stores: list[IndexStoreQueryPort] = [
            BM25IndexStoreQuery(bm25_dir_path, weight=0.65),
            ChromaIndexStoreQuery(
                chroma_dir_path,
                embedding_model_name,
                enable=manifest.with_semantic,
                weight=0.35,
            ),
        ]

        retriever = Retriever(
            index_store_registry=IndexStoreRegistry(*query_stores),
            chunks_loader=ChunksJSONLoader(chunks_file_path),
        )

        translator = TextGenerationTranslatorLLM()

        return retriever, translator
