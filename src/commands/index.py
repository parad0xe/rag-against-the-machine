from __future__ import annotations

import logging
from pathlib import Path

from pydantic import PositiveInt

from src.infrastructure.indexers.indexer import Indexer
from src.infrastructure.indexers.stores.bm25 import BM25IndexStore
from src.infrastructure.indexers.stores.chroma import ChromaIndexStore
from src.infrastructure.indexers.stores.raw import RawChunkStore
from src.infrastructure.manifest.manager import ManifestManager
from src.utils.path_util import parse_extensions

logger = logging.getLogger(__file__)

BATCH_SIZE = 32


def entrypoint_index(
    repositories: list[Path] = [Path("vllm-0.10.1")],
    extensions: str = "*",
    chunk_size: PositiveInt = 2000,
    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    manifest_filepath: Path = Path("data/processed/manifest.json"),
    bm25_dirpath: Path = Path("data/processed/bm25_index"),
    chroma_dirpath: Path = Path("data/processed/chroma_index"),
    chunks_filepath: Path = Path("data/processed/chunks.json"),
    with_semantic: bool = False,
) -> None:
    exts: list[str] = parse_extensions(extensions)
    manifest_manager = ManifestManager(
        manifest_filepath,
        list(map(str, repositories)),
        embedding_model_name,
        chunk_size,
        identity=[embedding_model_name, chunk_size, with_semantic],
    )

    indexer = Indexer(
        manifest_manager,
        extensions=exts,
        stores=[
            BM25IndexStore(bm25_dirpath, enable=True),
            ChromaIndexStore(
                chroma_dirpath,
                embedding_model_name,
                batch_size=BATCH_SIZE,
                enable=with_semantic,
            ),
            RawChunkStore(chunks_filepath, True),
        ],
    )
    indexer.sync()
    for repository in manifest_manager.manifest.repositories:
        indexer.index(Path(repository))
    indexer.commit()

    manifest_manager.commit()
