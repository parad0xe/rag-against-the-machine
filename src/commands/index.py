from __future__ import annotations

import logging
from pathlib import Path

from pydantic import PositiveInt

from src.infrastructure.document.stores.bm25.sync import BM25IndexStoreSync
from src.infrastructure.document.stores.chroma.sync import ChromaIndexStoreSync
from src.infrastructure.document.stores.raw.sync import RawIndexStoreSync
from src.infrastructure.document.stores.registry import IndexStoreSyncRegistry
from src.infrastructure.indexer import Indexer
from src.infrastructure.manifest.manager import ManifestManager
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

    manifest_manager = ManifestManager.load(
        file_path=manifest_file_path,
        repositories=repositories,
        embedding_model_name=embedding_model_name,
        chunk_size=chunk_size,
        extensions=parsed_extensions,
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
    )

    for repository in repositories:
        indexer.index(repository)

    indexer.commit()
