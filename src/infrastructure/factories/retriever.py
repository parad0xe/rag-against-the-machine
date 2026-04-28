from pathlib import Path

from src.application.services.retriever import Retriever
from src.domain.exceptions.storage import StorageFileNotFoundError
from src.infrastructure.index_stores.bm25.query import BM25IndexStoreQuery
from src.infrastructure.index_stores.chroma.query import (
    ChromaIndexStoreQuery,
)
from src.infrastructure.index_stores.registry import IndexStoreQueryRegistry
from src.infrastructure.loaders.chunks import ChunksLoader
from src.infrastructure.loaders.manifest import ManifestJSONLoader
from src.infrastructure.translators.hugging_face import HuggingFaceTranslator


class RetrieverFactory:
    @staticmethod
    def build(
        bm25_dir_path: Path,
        chroma_dir_path: Path,
        chunks_file_path: Path,
        manifest_file_path: Path,
        embedding_model_name: str,
    ) -> tuple[Retriever, HuggingFaceTranslator]:

        manifest = ManifestJSONLoader().load(manifest_file_path)
        if not manifest:
            raise StorageFileNotFoundError(manifest_file_path)

        translator = HuggingFaceTranslator()

        retriever = Retriever(
            index_store_registry=IndexStoreQueryRegistry(
                BM25IndexStoreQuery(bm25_dir_path, weight=0.65),
                ChromaIndexStoreQuery(
                    chroma_dir_path,
                    embedding_model_name,
                    enable=manifest.with_semantic,
                    weight=0.35,
                ),
            ),
            chunks_loader=ChunksLoader(chunks_file_path),
        )

        return retriever, translator
