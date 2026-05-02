import logging
from pathlib import Path

from pydantic import (
    PositiveInt,
    validate_call,
)
from rich import get_console
from rich.table import Table

from src.factories.retriever import (
    RetrieverFactory,
)

logger = logging.getLogger(__file__)


@validate_call()
def entrypoint_search(
    original_query: str,
    bm25_dir_path: Path,
    chroma_dir_path: Path,
    chunks_file_path: Path,
    manifest_file_path: Path,
    embedding_model_name: str,
    k: PositiveInt,
) -> None:
    """
    Executes a search and displays the retrieved sources in a table.

    Args:
        original_query: The user's search query.
        bm25_dir_path: Path to the BM25 index directory.
        chroma_dir_path: Path to the ChromaDB directory.
        chunks_file_path: Path to the chunks JSON file.
        manifest_file_path: Path to the manifest JSON file.
        embedding_model_name: Name of the embedding model to use.
        k: Number of sources to retrieve.
    """
    console = get_console()

    console.print()
    console.rule("[bold blue]Search[/]", style="blue")
    console.print(f"\n[bold]Query:[/] [cyan]{original_query}[/]\n")
    console.print()

    console.print("[bold cyan][1/2][/] Initializing environment")
    with console.status(
        "Loading models and parsing data",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        retriever, _ = RetrieverFactory.build(
            bm25_dir_path,
            chroma_dir_path,
            chunks_file_path,
            manifest_file_path,
            embedding_model_name,
        )
    console.print("[bold green][ OK ][/] Models and data loaded.\n")

    console.print("[bold cyan][2/2][/] Executing search query")
    with console.status(
        "Searching for relevant documents",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        result, _, _ = retriever.search(
            original_query=original_query,
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

    console.rule("[bold green]Search completed[/]", style="blue")
    console.print()
