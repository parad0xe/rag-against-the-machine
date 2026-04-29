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

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

logger = logging.getLogger(__file__)


class App:
    def __init__(self) -> None:
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    def index(
        self,
        path: str = str(settings.repo_path),
        extensions: str | tuple[str] = "*",
        max_chunk_size: int = settings.max_chunk_size,
        semantic: bool = False,
        verbose: int = 0,
    ) -> None:
        self._prepare(verbose)

        if not isinstance(extensions, str):
            if isinstance(extensions, Iterable):
                extensions = ",".join(extensions)

        entrypoint_index(
            repositories=[Path(path)],
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
        self._prepare(verbose)

        entrypoint_search(
            original_query=query,
            k=k,
            bm25_dir_path=settings.bm25_dir,
            chunks_file_path=settings.chunks_path,
            chroma_dir_path=settings.chroma_dir,
            manifest_file_path=settings.manifest_path,
            embedding_model_name=settings.embedding_model,
        )

    def search_dataset(
        self,
        save_dir_path: str = str(settings.search_output),
        dataset_file_path: str = str(settings.unanswered_path),
        k: int = settings.default_k,
        verbose: int = 0,
    ) -> None:
        self._prepare(verbose)

        entrypoint_search_dataset(
            dataset_file_path=Path(dataset_file_path),
            save_dir_path=Path(save_dir_path),
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
        details: bool = False,
        verbose: int = 0,
    ) -> None:
        self._prepare(verbose)

        entrypoint_answer(
            original_query=query,
            k=k,
            bm25_dir_path=settings.bm25_dir,
            chunks_file_path=settings.chunks_path,
            chroma_dir_path=settings.chroma_dir,
            manifest_file_path=settings.manifest_path,
            embedding_model_name=settings.embedding_model,
            with_details=details,
        )

    def answer_dataset(
        self,
        save_dir_path: str = str(settings.answer_output),
        dataset_file_path: str = str(settings.answered_path),
        k: int = settings.default_k,
        verbose: int = 0,
    ) -> None:
        self._prepare(verbose)

        entrypoint_answer_dataset(
            dataset_file_path=Path(dataset_file_path),
            save_dir_path=Path(save_dir_path),
            k=k,
            bm25_dir_path=settings.bm25_dir,
            chunks_file_path=settings.chunks_path,
            chroma_dir_path=settings.chroma_dir,
            manifest_file_path=settings.manifest_path,
            embedding_model_name=settings.embedding_model,
        )

    def evaluate(
        self,
        dataset_file_path: str = str(settings.answered_path),
        predictions_file_path: str = str(settings.search_output),
        ks: str | tuple[int, ...] = (1, 3, 5, 10),
        verbose: int = 0,
    ) -> None:
        self._prepare(verbose)

        if isinstance(ks, str):
            ks = tuple(int(k.strip()) for k in ks.split(",") if k.strip())

        entrypoint_evaluate(
            dataset_file_path=Path(dataset_file_path),
            predictions_file_path=Path(predictions_file_path),
            ks=ks,
        )

    def manifest_stats(
        self,
        path: str = str(settings.manifest_path),
        all: bool = False,
        verbose: int = 0,
    ) -> None:
        self._prepare(verbose)
        entrypoint_manifest_stats(manifest_file_path=Path(path), all=all)

    def _prepare(self, verbose: int) -> None:
        v = TypeAdapter(NonNegativeInt).validate_python(verbose)
        LoggingSystem.global_setup(v)


def main() -> None:
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
