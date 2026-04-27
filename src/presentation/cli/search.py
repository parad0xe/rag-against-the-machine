import logging
from pathlib import Path

from pydantic import (
    validate_call,
)
from rich.console import Console
from rich.table import Table

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
    console = Console()

    console.print()
    console.rule("[bold blue]Knowledge Base Search[/]", style="blue")
    console.print(f"\n[bold]Query:[/] [cyan]{original_query}[/]\n")

    console.print("[bold cyan][1/2][/] Initializing environment...")
    with console.status(
        "Loading models and parsing data...",
        spinner="dots",
        spinner_style="bold magenta",
    ):
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
    console.print("[bold green][ OK ][/] Models and data loaded.\n")

    console.print("[bold cyan][2/2][/] Executing search query...")
    with console.status(
        "Searching for relevant documents...",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        result = retriever.search(
            original_query=original_query,
            translator=translator,
            k=k,
        )
    console.print("[bold green][ OK ][/] Search completed.\n")

    if not result or not result.retrieved_sources:
        console.print(
            "[bold yellow]No relevant documents found for this query.[/]\n"
        )
        console.rule(style="blue")
        console.print()
        return

    table = Table(
        title=f"Results (Question ID: {result.question_id})",
        title_justify="left",
        title_style="dim",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
        expand=True,
    )

    table.add_column("Rank", justify="center", style="bold yellow", width=6)
    table.add_column("File Path", style="bold white")
    table.add_column("Position (Chars)", justify="right", style="cyan")

    for i, source in enumerate(result.retrieved_sources):
        table.add_row(
            f"#{i + 1}",
            source.file_path,
            f"{source.first_character_index} ➔ {source.last_character_index}",
        )

    console.print(table)
    console.print(
        "[dim]Total sources retrieved: "
        f"{len(result.retrieved_sources)}[/dim]\n"
    )
    console.rule(style="blue")
    console.print()
