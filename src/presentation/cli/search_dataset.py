import logging
import time
from pathlib import Path

from pydantic import (
    validate_call,
)
from rich import get_console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.domain.models.inference import StudentSearchResults
from src.factories.retriever import RetrieverFactory
from src.infrastructure.dataset.reader import RagDatasetJSONReader
from src.utils.file import file_write_json

logger = logging.getLogger(__file__)


@validate_call()
def entrypoint_search_dataset(
    dataset_file_path: Path,
    save_dir_path: Path,
    k: int,
    bm25_dir_path: Path,
    chroma_dir_path: Path,
    chunks_file_path: Path,
    manifest_file_path: Path,
    embedding_model_name: str,
) -> None:
    console = get_console()

    console.print()
    console.rule("[bold blue]Search dataset[/]", style="blue")
    console.print()

    console.print("[bold cyan][1/3][/] Initializing environment...")
    with console.status(
        "Loading models and parsing data...",
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

        dataset = RagDatasetJSONReader().read(dataset_file_path)

    console.print("[bold green][ OK ][/] Models and data loaded.\n")

    console.print(
        f"[bold cyan][2/3][/] Processing {len(dataset.rag_questions)} "
        "questions..."
    )

    results_stream = retriever.search_dataset_stream(
        dataset=dataset,
        k=k,
    )

    start_time = time.perf_counter()

    student = StudentSearchResults(k=k)
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
        task_id = progress.add_task(
            "Searching dataset...", total=len(dataset.rag_questions)
        )

        for result, _ in results_stream:
            progress.update(
                task_id,
                description=(
                    f"[bold cyan]Searching:[/] [dim]{result.question_id}[/]"
                ),
            )

            student.search_results.append(result)
            progress.advance(task_id)

        progress.update(task_id, description="[bold green]Search completed[/]")

    elapsed_time = time.perf_counter() - start_time

    console.print(
        f"[bold green][ OK ][/] Dataset search finished in "
        f"[bold yellow]{elapsed_time:.2f}s[/].\n"
    )

    console.print("[bold cyan][3/3][/] Saving results to disk...")
    with console.status(
        "Writing JSON file...",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        file_write_json(save_dir_path, student.model_dump_json(indent=2))

    console.print(
        f"[bold green][ OK ][/] Results successfully saved to "
        f"[bold white]{save_dir_path}[/]\n"
    )

    console.rule("[bold green]Search dataset completed[/]", style="blue")
    console.print()
