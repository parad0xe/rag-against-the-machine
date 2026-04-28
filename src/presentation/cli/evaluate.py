import json
from pathlib import Path
from typing import cast

from pydantic import validate_call
from rich import get_console
from rich.table import Table

from src.application.services.evaluator import Evaluator
from src.domain.exceptions.storage import StorageFileNotFoundError
from src.domain.models.question import AnsweredQuestion
from src.domain.models.student import (
    StudentSearchResults,
)
from src.infrastructure.loaders.rag_dataset import RagDatasetJSONLoader


@validate_call()
def entrypoint_evaluate(
    dataset_file_path: Path,
    predictions_file_path: Path,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> None:
    console = get_console()
    console.print("\n[bold blue]--- Evaluation ---[/]\n")

    dataset = RagDatasetJSONLoader().load(dataset_file_path)
    if not dataset:
        raise StorageFileNotFoundError(dataset_file_path)

    if not predictions_file_path.exists():
        raise StorageFileNotFoundError(predictions_file_path)

    with open(predictions_file_path, "r", encoding="utf-8") as f:
        student_data = StudentSearchResults.model_validate(json.load(f))

    evaluator = Evaluator()

    valid_ks = sorted([k for k in ks if k <= student_data.k])
    if not valid_ks:
        valid_ks = [student_data.k]

    total_recalls = {k: 0.0 for k in valid_ks}
    evaluated_count = 0

    expected_map = {
        q.question_id: cast(AnsweredQuestion, q).sources
        for q in dataset.rag_questions
        if hasattr(q, "sources")
    }

    for result in student_data.search_results:
        expected_sources = expected_map.get(result.question_id)
        if expected_sources is None:
            continue

        for k in valid_ks:
            retrieved = result.retrieved_sources[:k]
            recall = evaluator.calculate_recall(
                retrieved=retrieved,
                expected=expected_sources,
            )
            total_recalls[k] += recall

        evaluated_count += 1

    table = Table(
        title="Evaluation Results",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Score", justify="right", style="bold yellow")

    for k in valid_ks:
        if evaluated_count > 0:
            final_score = total_recalls[k] / evaluated_count
        else:
            final_score = 0.0
        table.add_row(f"Recall@{k}", f"{final_score:.4f}")

    console.print(table)
    console.print(f"\n[dim]Questions evaluated: {evaluated_count}[/dim]\n")
