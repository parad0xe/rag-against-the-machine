import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable

import fire
from fire.core import FireExit
from pydantic import (
    NonNegativeInt,
    TypeAdapter,
    ValidationError,
)
from rich.console import Console
from tqdm import TqdmExperimentalWarning

from src.config import settings
from src.domain.exceptions.base import RagError
from src.domain.exceptions.schema import SchemaValidationError
from src.logging import LoggingSystem
from src.presentation.cli.answer import entrypoint_answer
from src.presentation.cli.answer_dataset import entrypoint_answer_dataset
from src.presentation.cli.evaluate import entrypoint_evaluate
from src.presentation.cli.index import entrypoint_index
from src.presentation.cli.manifest_stats import entrypoint_manifest_stats
from src.presentation.cli.search import entrypoint_search
from src.presentation.cli.search_dataset import entrypoint_search_dataset
from src.utils.file import ensure_valid_file_path

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

logger = logging.getLogger(__file__)


class App:
    """
    CLI application class that maps commands to their respective entrypoints.
    """

    def __init__(self) -> None:
        """Initializes the application and ensures data directories exist."""
        settings.processed_dir.mkdir(parents=True, exist_ok=True)

    def index(
        self,
        path: str = str(settings.repo_path),
        extensions: str | tuple[str] = "*",
        max_chunk_size: int = settings.max_chunk_size,
        semantic: bool = False,
        verbose: int = 0,
    ) -> None:
        """
        Indexes a repository.

        Args:
            path: Path to the repository.
            extensions: File extensions to index.
            max_chunk_size: Maximum size of text chunks.
            semantic: Whether to enable semantic search capabilities.
            verbose: Logging verbosity level (0-2).
        """
        self._prepare(verbose)

        if not isinstance(extensions, str):
            if isinstance(extensions, Iterable):
                extensions = ",".join(extensions)

        entrypoint_index(
            repositories=[Path(str(path))],
            bm25_dir_path=settings.bm25_dir,
            manifest_file_path=settings.manifest_path,
            chroma_dir_path=settings.chroma_dir,
            embedding_model_name=settings.embedding_model,
            chunks_file_path=settings.chunks_path,
            extensions=extensions,
            chunk_size=max_chunk_size,
            with_semantic=semantic,
        )

    def search(
        self,
        query: str,
        k: int = settings.default_k,
        verbose: int = 0,
    ) -> None:
        """
        Searches for relevant chunks.

        Args:
            query: The search query.
            k: Number of results to retrieve.
            verbose: Logging verbosity level (0-2).
        """
        self._prepare(verbose)

        if not query.strip():
            raise RagError("query is empty")

        entrypoint_search(
            original_query=str(query),
            k=k,
            bm25_dir_path=settings.bm25_dir,
            chunks_file_path=settings.chunks_path,
            chroma_dir_path=settings.chroma_dir,
            manifest_file_path=settings.manifest_path,
            embedding_model_name=settings.embedding_model,
        )

    def search_dataset(
        self,
        save_dir_path: str = str(settings.search_output_dir_path),
        dataset_file_path: str = str(settings.unanswered_path),
        k: int = settings.default_k,
        verbose: int = 0,
    ) -> None:
        """
        Performs search for a dataset of questions.

        Args:
            save_dir_path: Directory to save results.
            dataset_file_path: Path to the input dataset.
            k: Number of results per question.
            verbose: Logging verbosity level (0-2).
        """
        self._prepare(verbose)
        ensure_valid_file_path(dataset_file_path)

        entrypoint_search_dataset(
            dataset_file_path=Path(str(dataset_file_path)),
            save_dir_path=Path(str(save_dir_path)),
            k=k,
            bm25_dir_path=settings.bm25_dir,
            chunks_file_path=settings.chunks_path,
            chroma_dir_path=settings.chroma_dir,
            manifest_file_path=settings.manifest_path,
            embedding_model_name=settings.embedding_model,
        )

    def answer(
        self,
        query: str,
        k: int = settings.default_k,
        thinking: bool = False,
        verbose: int = 0,
    ) -> None:
        """
        Answers a question using RAG.

        Args:
            query: The question to answer.
            k: Number of chunks for context.
            thinking: Whether to show the reasoning process.
            verbose: Logging verbosity level (0-2).
        """
        self._prepare(verbose)

        if not query.strip():
            raise RagError("query is empty")

        entrypoint_answer(
            original_query=str(query),
            k=k,
            bm25_dir_path=settings.bm25_dir,
            chunks_file_path=settings.chunks_path,
            chroma_dir_path=settings.chroma_dir,
            manifest_file_path=settings.manifest_path,
            embedding_model_name=settings.embedding_model,
            thinking=thinking,
        )

    def answer_dataset(
        self,
        save_dir_path: str = str(settings.answer_output_dir_path),
        dataset_file_path: str = str(settings.answered_path),
        k: int = settings.default_k,
        thinking: bool = False,
        verbose: int = 0,
    ) -> None:
        """
        Answers a dataset of questions.

        Args:
            save_dir_path: Directory to save answers.
            dataset_file_path: Path to the input dataset.
            k: Number of chunks per context.
            thinking: Whether to show reasoning.
            verbose: Logging verbosity level (0-2).
        """
        self._prepare(verbose)
        ensure_valid_file_path(dataset_file_path)

        entrypoint_answer_dataset(
            dataset_file_path=Path(str(dataset_file_path)),
            save_dir_path=Path(str(save_dir_path)),
            k=k,
            bm25_dir_path=settings.bm25_dir,
            chunks_file_path=settings.chunks_path,
            chroma_dir_path=settings.chroma_dir,
            manifest_file_path=settings.manifest_path,
            embedding_model_name=settings.embedding_model,
            thinking=thinking,
        )

    def evaluate(
        self,
        predictions_file_path: str,
        dataset_file_path: str = str(settings.answered_path),
        ks: str | tuple[int, ...] = (1, 3, 5, 10),
        verbose: int = 0,
    ) -> None:
        """
        Evaluates search results.

        Args:
            predictions_file_path: Path to the results JSON.
            dataset_file_path: Path to ground truth.
            ks: List of k values for recall calculation.
            verbose: Logging verbosity level (0-2).
        """
        self._prepare(verbose)
        ensure_valid_file_path(predictions_file_path)
        ensure_valid_file_path(dataset_file_path)

        if isinstance(ks, str):
            try:
                ks = tuple(int(k.strip()) for k in ks.split(",") if k.strip())
            except ValueError:
                raise RagError(f"can't convert str {ks} to tuple of int.")
        elif isinstance(ks, int):
            ks = (ks,)

        entrypoint_evaluate(
            dataset_file_path=Path(str(dataset_file_path)),
            predictions_file_path=Path(str(predictions_file_path)),
            ks=ks,
        )

    def manifest_stats(
        self,
        path: str = str(settings.manifest_path),
        all: bool = False,
        verbose: int = 0,
    ) -> None:
        """
        Displays manifest statistics.

        Args:
            path: Path to the manifest file.
            all: Whether to show detailed extension stats.
            verbose: Logging verbosity level (0-2).
        """
        self._prepare(verbose)
        ensure_valid_file_path(path)

        entrypoint_manifest_stats(manifest_file_path=Path(path), all=all)

    def _prepare(self, verbose: int) -> None:
        """
        Configures logging and validates verbosity.

        Args:
            verbose: Logging verbosity level.
        """
        v = TypeAdapter(NonNegativeInt).validate_python(verbose)
        LoggingSystem.global_setup(v)


def main() -> None:
    """Entrypoint function for the CLI."""
    logger.info("Application starting")
    os.environ.setdefault("PAGER", "cat")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    error_console = Console(stderr=True)

    try:
        try:
            fire.Fire(App())
        except ValidationError as e:
            raise SchemaValidationError(e) from e
    except RagError as e:
        error_console.print(f"\n[bold red] {e.__class__.__name__}:[/] {e}")
        sys.exit(1)
    except FireExit:
        sys.exit(2)
    except Exception as e:
        error_console.print(f"\n[bold red] {e.__class__.__name__}:[/] {e}")
        error_console.print_exception(show_locals=False)
        sys.exit(3)


if __name__ == "__main__":
    main()
