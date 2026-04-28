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

from src.domain.exceptions.storage import StorageFileNotFoundError
from src.domain.models.answer import MinimalAnswer
from src.domain.models.student import StudentSearchResultsAndAnswer
from src.infrastructure.factories.retriever import RetrieverFactory
from src.infrastructure.llm.qwen import QwenLlm
from src.infrastructure.loaders.rag_dataset import RagDatasetJSONLoader
from src.utils.file import file_write_json

logger = logging.getLogger(__file__)

DEFAULT_MODEL_NAME: str = "Qwen/Qwen3-0.6B"


@validate_call()
def entrypoint_answer_dataset(
    dataset_file_path: Path,
    save_dir_path: Path,
    k: int,
    bm25_dir_path: Path,
    chroma_dir_path: Path,
    chunks_file_path: Path,
    manifest_file_path: Path,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> None:
    console = get_console()

    console.print("\n[bold blue]--- Dataset Search Processing ---[/]\n")

    console.print("[bold cyan][1/3][/] Initializing environment...")
    with console.status(
        "Loading models and parsing data...",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        retriever, translator = RetrieverFactory.build(
            bm25_dir_path,
            chroma_dir_path,
            chunks_file_path,
            manifest_file_path,
            embedding_model_name,
        )

        dataset = RagDatasetJSONLoader().load(dataset_file_path)
        if not dataset:
            raise StorageFileNotFoundError(dataset_file_path)

        llm = QwenLlm(model_name=DEFAULT_MODEL_NAME)

    console.print("[bold green][ OK ][/] Models and data loaded.\n")

    console.print(
        f"[bold cyan][2/3][/] Processing {len(dataset.rag_questions)} "
        "questions..."
    )

    results_stream = retriever.search_dataset_stream(
        dataset=dataset,
        translator=translator,
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
            "Processing dataset...", total=len(dataset.rag_questions)
        )

        for result, chunks in results_stream:
            progress.update(
                task_id,
                description=(
                    f"[bold cyan]Processing:[/] [dim]{result.question_id}[/]"
                ),
            )

            context: list[str] = []
            for index, chunk in enumerate(chunks):
                file_path = chunk.get("file_path")
                start_index = chunk.get("first_character_index")
                end_index = chunk.get("last_character_index")
                text_content = chunk.get("text")

                context_source = (
                    f"---  SOURCE #{index + 1} ---\n"
                    f"File: {file_path} (Chars: {start_index}-{end_index})\n"
                    f"Content: {text_content}"
                )
                context.append(context_source)

            answer_stream = llm.generate_answer(
                query=result.question,
                context="\n".join(context),
            )

            full_text = "".join(answer_stream)

            if "<think>" in full_text:
                parts = full_text.split("<think>", 1)
                after_think = parts[1]
                if "</think>" in after_think:
                    think_parts = after_think.split("</think>", 1)
                    full_text = think_parts[1].strip("\n")
            else:
                full_text = full_text.strip("\n")

            full_text = full_text.strip("\n")

            minimal_answer = MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=full_text,
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
