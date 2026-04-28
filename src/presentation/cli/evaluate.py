import json
from pathlib import Path
from typing import cast

from pydantic import validate_call
from rich import get_console
from rich.table import Table

from src.application.services.evaluator import Evaluator
from src.domain.exceptions.storage import StorageFileNotFoundError
from src.domain.models.dataset import AnsweredQuestion
from src.domain.models.inference import (
    StudentSearchResults,
)
from src.infrastructure.dataset.json_loader import RagDatasetJSONLoader


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

    dataset.rag_questions = dataset.rag_questions

    if not predictions_file_path.exists():
        raise StorageFileNotFoundError(predictions_file_path)

    with open(predictions_file_path, "r", encoding="utf-8") as f:
        student_data = StudentSearchResults.model_validate(json.load(f))

    evaluator = Evaluator()

    valid_ks = sorted([k for k in ks if k <= student_data.k])
    if not valid_ks:
        valid_ks = [student_data.k]

    total_recalls = {k: 0.0 for k in valid_ks}
    evaluated_sources = 0

    results_map = {res.question_id: res for res in student_data.search_results}

    for expected in dataset.rag_questions:
        expected_answer = cast(AnsweredQuestion, expected)
        result = results_map.get(expected_answer.question_id)

        for k in valid_ks:
            if result is None:
                recall = 0.0
            else:
                retrieved = result.retrieved_sources[:k]
                recall = evaluator.calculate_recall(
                    retrieved=retrieved,
                    expected=expected_answer.sources,
                )
            total_recalls[k] += recall

        evaluated_sources += len(expected_answer.sources)

    table = Table(
        title="Evaluation Results",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Score", justify="right", style="bold yellow")

    for k in valid_ks:
        if evaluated_sources > 0:
            final_score = total_recalls[k] / evaluated_sources
        else:
            final_score = 0.0
        table.add_row(f"Recall@{k}", f"{final_score:.4f}")

    console.print(table)
    console.print(
        f"\n[dim]Questions evaluated: {len(dataset.rag_questions)}[/dim]\n"
    )
