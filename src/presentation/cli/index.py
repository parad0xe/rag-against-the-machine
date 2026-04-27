from __future__ import annotations

import logging
from pathlib import Path

from pydantic import PositiveInt

from src.application.services.indexer import Indexer
from src.infrastructure.index_stores.bm25.sync import BM25IndexStoreSync
from src.infrastructure.index_stores.chroma.sync import ChromaIndexStoreSync
from src.infrastructure.index_stores.raw.sync import RawIndexStoreSync
from src.infrastructure.index_stores.registry import IndexStoreSyncRegistry
from src.infrastructure.loaders.document.document import DocumentLoader
from src.infrastructure.loaders.file import LocalFileLoader
from src.infrastructure.storage.manifest.json_storage import (
    ManifestJSONStorage,
)
from src.infrastructure.storage.manifest.manager import ManifestManager
from src.utils.common import parse_extensions

logger = logging.getLogger(__file__)

BATCH_SIZE = 32


def entrypoint_index(
    repositories: list[Path],
    manifest_file_path: Path,
    bm25_dir_path: Path,
    chroma_dir_path: Path,
    chunks_file_path: Path,
    extensions: str = "*",
    chunk_size: PositiveInt = 2000,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    with_semantic: bool = False,
) -> None:
    repositories = list({repo.resolve() for repo in repositories})
    parsed_extensions: list[str] = parse_extensions(extensions)

    logger.info(f"loaded extensions: {parsed_extensions}")

    # TD: Systeme a ameliorer (trop verbeux)
    manifest_manager = ManifestManager(
        file_path=manifest_file_path,
        manifest_storage=ManifestJSONStorage(),
        extensions=parsed_extensions,
        repositories=repositories,
        embedding_model_name=embedding_model_name,
        chunk_size=chunk_size,
        with_semantic=with_semantic,
        fingerprint_seed=[embedding_model_name, chunk_size],
    )

    indexer = Indexer(
        manifest_manager=manifest_manager,
        extensions=parsed_extensions,
        index_store_registry=IndexStoreSyncRegistry(
            BM25IndexStoreSync(bm25_dir_path),
            ChromaIndexStoreSync(
                chroma_dir_path,
                embedding_model_name,
                batch_size=BATCH_SIZE,
                addition_enable=with_semantic,
            ),
            RawIndexStoreSync(chunks_file_path),
        ),
        file_loader=LocalFileLoader(),
        document_loader=DocumentLoader(),
    )

    for repository in repositories:
        indexer.index(repository)

    indexer.commit()
