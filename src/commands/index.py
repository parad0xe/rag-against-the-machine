from __future__ import annotations

import logging
from pathlib import Path

from pydantic import PositiveInt

from src.infrastructure.document.stores.bm25.sync import BM25SyncIndexStore
from src.infrastructure.document.stores.chroma.sync import ChromaSyncIndexStore
from src.infrastructure.document.stores.raw.sync import RawSyncIndexStore
from src.infrastructure.document.stores.registry import SyncIndexStoreRegistry
from src.infrastructure.indexer import Indexer
from src.infrastructure.repositories.manifest import ManifestRepository
from src.utils.path_util import parse_extensions

logger = logging.getLogger(__file__)

BATCH_SIZE = 32


def entrypoint_index(
    repositories: list[Path],
    manifest_filepath: Path,
    bm25_dirpath: Path,
    chroma_dirpath: Path,
    chunks_filepath: Path,
    extensions: str = "*",
    chunk_size: PositiveInt = 2000,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    with_semantic: bool = False,
) -> None:
    repositories = list({repo.resolve() for repo in repositories})
    exts: list[str] = parse_extensions(extensions)

    manifest_repository = ManifestRepository(
        manifest_filepath,
        repositories,
        embedding_model_name,
        chunk_size,
        identity=[embedding_model_name, chunk_size],
    )

    indexer = Indexer(
        manifest_repository=manifest_repository,
        extensions=exts,
        index_store_registry=SyncIndexStoreRegistry(
            BM25SyncIndexStore(bm25_dirpath, enable=True),
            ChromaSyncIndexStore(
                chroma_dirpath,
                embedding_model_name,
                batch_size=BATCH_SIZE,
                enable=with_semantic,
            ),
            RawSyncIndexStore(chunks_filepath, True),
        ),
    )
    indexer.sync()

    for repository in manifest_repository.manifest.repositories:
        indexer.index(repository)

    indexer.commit()
    manifest_repository.commit()
