import logging
import time
from pathlib import Path

from pydantic import (
    validate_call,
)
from rich.console import Console
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

from src.application.services.retriever import Retriever
from src.domain.exceptions.storage import StorageFileNotFoundError
from src.domain.models.student import StudentSearchResults
from src.infrastructure.index_stores.bm25.query import BM25IndexStoreQuery
from src.infrastructure.index_stores.chroma.query import (
    ChromaIndexStoreQuery,
)
from src.infrastructure.index_stores.registry import IndexStoreQueryRegistry
from src.infrastructure.loaders.chunks import ChunksLoader
from src.infrastructure.loaders.manifest import ManifestJSONLoader
from src.infrastructure.loaders.rag_dataset import RagDatasetJSONLoader
from src.infrastructure.translators.hugging_face import HuggingFaceTranslator
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
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> None:
    console = Console()

    console.print("\n[bold blue]--- Dataset Search Processing ---[/]\n")

    console.print("[bold cyan][1/3][/] Initializing environment...")
    with console.status(
        "Loading models and parsing data...",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        manifest = ManifestJSONLoader().load(manifest_file_path)
        if not manifest:
            raise StorageFileNotFoundError(manifest_file_path)

        dataset = RagDatasetJSONLoader().load(dataset_file_path)
        if not dataset:
            raise StorageFileNotFoundError(dataset_file_path)

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

    console.print(
        f"[bold cyan][2/3][/] Processing {len(dataset.rag_questions)} "
        "questions..."
    )
    student = StudentSearchResults(k=k)

    results_stream = retriever.search_dataset_stream(
        dataset=dataset,
        translator=translator,
        k=k,
    )

    start_time = time.perf_counter()

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

        for result in results_stream:
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
