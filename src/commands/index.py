from __future__ import annotations

import logging
from pathlib import Path

from pydantic import PositiveInt

from src.infrastructure.document.stores.bm25.sync import BM25IndexStoreSync
from src.infrastructure.document.stores.chroma.sync import ChromaIndexStoreSync
from src.infrastructure.document.stores.raw.sync import RawIndexStoreSync
from src.infrastructure.document.stores.registry import IndexStoreSyncRegistry
from src.infrastructure.indexer import Indexer
from src.infrastructure.repositories.manifest import ManifestRepository
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

    manifest_repository = ManifestRepository(
        manifest_file_path,
        repositories,
        embedding_model_name,
        chunk_size,
        fingerprint_seed=[embedding_model_name, chunk_size],
    )

    indexer = Indexer(
        manifest_repository=manifest_repository,
        extensions=parse_extensions(extensions),
        index_store_registry=IndexStoreSyncRegistry(
            BM25IndexStoreSync(bm25_dir_path, enable=True),
            ChromaIndexStoreSync(
                chroma_dir_path,
                embedding_model_name,
                batch_size=BATCH_SIZE,
                enable=with_semantic,
            ),
            RawIndexStoreSync(chunks_file_path, True),
        ),
    )
    indexer.sync()

    for repository in manifest_repository.manifest.repositories:
        indexer.index(repository)

    indexer.commit()
    manifest_repository.commit()
