import logging
from pathlib import Path

from pydantic import (
    validate_call,
)

from src.application.services.retriever import Retriever
from src.infrastructure.index_stores.bm25.query import BM25IndexStoreQuery
from src.infrastructure.index_stores.chroma.query import (
    ChromaIndexStoreQuery,
)
from src.infrastructure.index_stores.registry import IndexStoreQueryRegistry
from src.infrastructure.loaders.chunks import ChunksLoader
from src.infrastructure.loaders.manifest import ManifestJSONLoader
from src.infrastructure.translators.hugging_face import HuggingFaceTranslator
from src.presentation.api.console.minimal_search import (
    MinimalSearchDisplayConsole,
)
from src.utils.file import ensure_valid_dir_path

logger = logging.getLogger(__file__)


@validate_call()
def entrypoint_search(
    original_query: str,
    bm25_dir_path: Path,
    chroma_dir_path: Path,
    chunks_file_path: Path,
    manifest_file_path: Path,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    k: int = 10,
) -> None:
    ensure_valid_dir_path(bm25_dir_path)
    ensure_valid_dir_path(chroma_dir_path)

    display = MinimalSearchDisplayConsole()
    display.title(original_query)

    translated_query = HuggingFaceTranslator().translate_to_english(
        original_query
    )

    if original_query != translated_query:
        display.subquery(translated_query)

    manifest = ManifestJSONLoader().load(manifest_file_path)

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

    with display.loader():
        result = retriever.search(
            original_query=original_query,
            tranlated_query=translated_query,
            k=k,
        )

    if not result:
        display.noresult()
        return

    display.results(result)
