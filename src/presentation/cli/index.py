from __future__ import annotations

import logging
import time
from pathlib import Path

from pydantic import Field, PositiveInt, validate_call
from rich import get_console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.application.ports.index_store import IndexStoreSyncPort
from src.application.services.indexer import IndexerService
from src.config import settings
from src.domain.exceptions.document import NoDocumentError
from src.infrastructure.document.loader import DocumentLoader
from src.infrastructure.file.reader import LocalFileReader
from src.infrastructure.index_stores.bm25.sync import BM25IndexStoreSync
from src.infrastructure.index_stores.chroma.sync import ChromaIndexStoreSync
from src.infrastructure.index_stores.raw.sync import RawIndexStoreSync
from src.infrastructure.index_stores.registry import IndexStoreRegistry
from src.infrastructure.manifest.manager import ManifestManager
from src.infrastructure.manifest.storage import ManifestJSONStorage
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
    chunk_size: PositiveInt = Field(2000, gt=200, le=2000),
) -> None:
    console = get_console()

    console.print()
    console.rule("[bold blue]Indexing[/]", style="blue")
    console.print()

    parsed_extensions: list[str] = parse_extensions(extensions)

    start_time = time.perf_counter()

    console.print("[bold cyan][1/3][/] Initializing manifest and stores")
    with console.status(
        "Loading configurations",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        manifest_manager = ManifestManager(
            file_path=manifest_file_path,
            storage=ManifestJSONStorage(),
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

        indexer = IndexerService(
            manifest_manager=manifest_manager,
            extensions=parsed_extensions,
            index_store_registry=IndexStoreRegistry(*sync_stores),
            file_loader=LocalFileReader(),
            document_loader=DocumentLoader(),
        )
    console.print("[bold green][ OK ][/] Initialization complete.\n")

    console.print(
        f"[bold cyan][2/3][/] Indexing {len(repositories)} repositories "
        f"(extensions: {', '.join(parsed_extensions)})"
    )

    with Progress(
        SpinnerColumn(spinner_name="dots", style="bold magenta"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for repository in repositories:
            task_id = progress.add_task(f"Scanning {repository.name}", total=1)

            for total_files, current_path in indexer.index(repository):
                progress.update(task_id, total=total_files)
                progress.update(
                    task_id,
                    description=(
                        f"[bold cyan]Parsing:[/] [dim]{current_path.name}[/]"
                    ),
                )
                progress.advance(task_id)

            progress.update(
                task_id,
                description=f"[bold green]{repository.name} indexed[/]",
            )
        progress.update(task_id, description="[bold green]Files scanned[/]")

    if not indexer.indexed_document_count:
        raise NoDocumentError()

    console.print(
        "[bold green][ OK ][/] Repositories scanned "
        f"({indexer.indexed_document_count} indexed documents)\n"
    )

    console.print("[bold cyan][3/3][/] Committing changes to stores")

    with Progress(
        SpinnerColumn(spinner_name="dots", style="bold magenta"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        active_tasks: dict[str, TaskID] = {}

        for store_name, current, total, desc in indexer.commit():
            if store_name not in active_tasks:
                active_tasks[store_name] = progress.add_task(
                    f"[bold cyan]{store_name}:[/] [dim]Starting...[/]",
                    total=total,
                )

            task_id = active_tasks[store_name]

            progress.update(
                task_id,
                total=total,
                completed=current,
                description=f"[bold cyan]{store_name}:[/] [dim]{desc}[/]",
            )

            if current >= total:
                progress.update(
                    task_id,
                    description=(
                        f"[bold green]{store_name}:[/] [dim]Synchronized[/]"
                    ),
                )

    console.print("\n[bold blue]--- Sync summary ---[/]")
    table = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("Store", style="bold white")
    table.add_column("Added Docs", justify="right", style="bold green")
    table.add_column("Added Chunks", justify="right", style="bold green")
    table.add_column("Deleted Chunks", justify="right", style="bold red")

    for store_name, stats in indexer.commit_summary.items():
        table.add_row(
            store_name,
            str(stats["added_docs"]),
            str(stats["added_chunks"]),
            str(stats["deleted_chunks"]),
        )

    console.print(table)

    elapsed_time = time.perf_counter() - start_time

    console.print(
        f"\n[bold green][ OK ][/] Indexes successfully updated in "
        f"[bold yellow]{elapsed_time:.2f}s[/].\n"
    )

    console.rule("[bold green]Indexing completed[/]", style="blue")
    console.print()
