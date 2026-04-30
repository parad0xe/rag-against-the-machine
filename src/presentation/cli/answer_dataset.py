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

from src.domain.models.inference import (
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from src.factories.retriever import RetrieverFactory
from src.infrastructure.dataset.reader import RagDatasetJSONReader
from src.utils.file import file_write_json
from src.utils.format import build_context_from_chunks, parse_llm_thought

logger = logging.getLogger(__file__)


@validate_call()
def entrypoint_answer_dataset(
    dataset_file_path: Path,
    save_dir_path: Path,
    bm25_dir_path: Path,
    chroma_dir_path: Path,
    chunks_file_path: Path,
    manifest_file_path: Path,
    k: int,
    embedding_model_name: str,
) -> None:
    console = get_console()

    console.print()
    console.rule("[bold blue]Answer dataset[/]", style="blue")
    console.print()

    console.print("[bold cyan][1/3][/] Initializing environment")
    with console.status(
        "Loading models and parsing data",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        retriever, llm = RetrieverFactory.build(
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
        "questions"
    )

    results_stream = retriever.search_dataset_stream(
        dataset=dataset,
        k=k,
    )

    start_time = time.perf_counter()

    student = StudentSearchResultsAndAnswer(k=k)

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
            "Processing dataset", total=len(dataset.rag_questions)
        )

        for result, chunks in results_stream:
            progress.update(
                task_id,
                description=(
                    f"[bold cyan]Processing:[/] [dim]{result.question_id}[/]"
                ),
            )

            answer_stream = llm.generate_answer(
                query=result.question,
                context=build_context_from_chunks(chunks),
            )

            full_text = "".join(answer_stream)
            _, final_text = parse_llm_thought(full_text)

            minimal_answer = MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=final_text,
            )

            student.search_results.append(minimal_answer)

            file_write_json(save_dir_path, student.model_dump_json(indent=2))

            progress.advance(task_id)

        progress.update(
            task_id, description="[bold green]Processing completed[/]"
        )

    elapsed_time = time.perf_counter() - start_time

    console.print(
        f"[bold green][ OK ][/] Dataset processing finished in "
        f"[bold yellow]{elapsed_time:.2f}s[/].\n"
    )

    console.rule("[bold green]Answer dataset completed[/]", style="blue")
    console.print()
