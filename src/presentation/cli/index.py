from __future__ import annotations

import logging
import time
from pathlib import Path

from pydantic import Field, PositiveInt, validate_call
from rich import get_console

from src.application.ports.index_store import IndexStoreSyncPort
from src.application.services.indexer import Indexer
from src.config import settings
from src.domain.exceptions.document import NoDocumentError
from src.infrastructure.document.loader import DocumentLoader
from src.infrastructure.file.reader import LocalFileReader
from src.infrastructure.index_stores.bm25.sync import BM25IndexStoreSync
from src.infrastructure.index_stores.chroma.sync import ChromaIndexStoreSync
from src.infrastructure.index_stores.raw.sync import RawIndexStoreSync
from src.infrastructure.index_stores.registry import IndexStoreRegistry
from src.infrastructure.manifest.manager import ManifestManager
from src.infrastructure.manifest.repository import ManifestJSONRepository
from src.utils.common import parse_extensions

logger = logging.getLogger(__file__)


@validate_call()
def entrypoint_index(
    repositories: list[Path],
    manifest_file_path: Path,
    bm25_dir_path: Path,
    chroma_dir_path: Path,
    chunks_file_path: Path,
    extensions: str,
    embedding_model_name: str,
    with_semantic: bool,
    chunk_size: PositiveInt = Field(2000, le=2000),
) -> None:
    console = get_console()

    console.print("\n[bold blue]--- Indexing ---[/]\n")

    repositories = list({repo.resolve() for repo in repositories})
    parsed_extensions: list[str] = parse_extensions(extensions)

    start_time = time.perf_counter()

    console.print("[bold cyan][1/3][/] Initializing manifest and stores...")
    with console.status(
        "Loading configurations...",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        manifest_manager = ManifestManager(
            file_path=manifest_file_path,
            manifest_repository=ManifestJSONRepository(),
            extensions=parsed_extensions,
            repositories=repositories,
            embedding_model_name=embedding_model_name,
            chunk_size=chunk_size,
            with_semantic=with_semantic,
            fingerprint_seed=[embedding_model_name, chunk_size],
        )

        sync_stores: list[IndexStoreSyncPort] = [
            BM25IndexStoreSync(bm25_dir_path),
            ChromaIndexStoreSync(
                chroma_dir_path,
                embedding_model_name,
                batch_size=settings.index_batch_size,
                addition_enable=with_semantic,
            ),
            RawIndexStoreSync(chunks_file_path),
        ]

        indexer = Indexer(
            manifest_manager=manifest_manager,
            extensions=parsed_extensions,
            index_store_registry=IndexStoreRegistry(*sync_stores),
            file_loader=LocalFileReader(),
            document_loader=DocumentLoader(),
        )
    console.print("[bold green][ OK ][/] Initialization complete.\n")

    console.print(
        f"[bold cyan][2/3][/] Indexing {len(repositories)} repositories "
        f"(extensions: {', '.join(parsed_extensions)})..."
    )
    for repository in repositories:
        indexer.index(repository)

    if not indexer.founded_documents:
        raise NoDocumentError()

    console.print("[bold green][ OK ][/] Repositories scanned and indexed.\n")

    console.print("[bold cyan][3/3][/] Committing changes to stores...")
    indexer.commit()

    elapsed_time = time.perf_counter() - start_time

    console.print(
        f"\n[bold green][ OK ][/] Indexes successfully updated in "
        f"[bold yellow]{elapsed_time:.2f}s[/].\n"
    )
