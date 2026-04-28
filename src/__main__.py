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
from tqdm import TqdmExperimentalWarning

from src.domain.exceptions.base import RagError
from src.domain.exceptions.schema import SchemaValidationError
from src.logging import LoggingSystem
from src.presentation.cli.answer import entrypoint_answer
from src.presentation.cli.answer_dataset import entrypoint_answer_dataset
from src.presentation.cli.evaluate import entrypoint_evaluate
from src.presentation.cli.index import entrypoint_index
from src.presentation.cli.search import entrypoint_search
from src.presentation.cli.search_dataset import entrypoint_search_dataset

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

logger = logging.getLogger(__file__)

DEFAULT_REPOSITORY_DIRPATH: str = "data/raw/vllm-0.10.1"

PROCESSED_DIR_PATH: Path = Path("data/processed")
BM25_DIRPATH: Path = PROCESSED_DIR_PATH / "bm25_index"
CHROMA_DIRPATH: Path = PROCESSED_DIR_PATH / "chroma_index"
MANIFEST_FILEPATH: Path = PROCESSED_DIR_PATH / "manifest.json"
CHUNK_FILEPATH: Path = PROCESSED_DIR_PATH / "chunks.json"
EMBEDDING_LLM_MODEL: str = "all-MiniLM-L6-v2"

DATASET_DIR_PATH: Path = Path("data/datasets")
UNANSWERED_FILEPATH: Path = (
    DATASET_DIR_PATH / "UnansweredQuestions/dataset_docs_public.json"
)
ANSWERED_FILEPATH: Path = (
    DATASET_DIR_PATH / "AnsweredQuestions/dataset_docs_public.json"
)

DATASET_OUTPUT: Path = Path("data/output")
DATASET_SEARCH_OUTPUT: Path = DATASET_OUTPUT / "search_results.json"
DATASET_ANSWER_OUTPUT: Path = DATASET_OUTPUT / "answer_results.json"


class App:
    def __init__(self) -> None:
        os.makedirs(PROCESSED_DIR_PATH, exist_ok=True)

    def index(
        self,
        path: str = DEFAULT_REPOSITORY_DIRPATH,
        extensions: str | tuple[str] = "*",
        max_chunk_size: int = 2000,
        semantic: bool = False,
        verbose: int = 0,
    ) -> None:
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)

        if not isinstance(extensions, str):
            if isinstance(extensions, Iterable):
                extensions = ",".join(extensions)

        self._init_logging(verbose)

        entrypoint_index(
            repositories=[Path(path)],
            bm25_dir_path=BM25_DIRPATH,
            manifest_file_path=MANIFEST_FILEPATH,
            chroma_dir_path=CHROMA_DIRPATH,
            embedding_model_name=EMBEDDING_LLM_MODEL,
            chunks_file_path=CHUNK_FILEPATH,
            extensions=extensions,
            chunk_size=max_chunk_size,
            with_semantic=semantic,
        )

    def search(
        self,
        query: str,
        k: int = 10,
        verbose: int = 0,
    ) -> None:
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)

        self._init_logging(verbose)

        entrypoint_search(
            original_query=query,
            k=k,
            bm25_dir_path=BM25_DIRPATH,
            chunks_file_path=CHUNK_FILEPATH,
            chroma_dir_path=CHROMA_DIRPATH,
            manifest_file_path=MANIFEST_FILEPATH,
            embedding_model_name=EMBEDDING_LLM_MODEL,
        )

    def search_dataset(
        self,
        save_dir_path: str = str(DATASET_SEARCH_OUTPUT),
        dataset_file_path: str = str(UNANSWERED_FILEPATH),
        k: int = 10,
        verbose: int = 0,
    ) -> None:
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)

        self._init_logging(verbose)

        entrypoint_search_dataset(
            dataset_file_path=Path(dataset_file_path),
            save_dir_path=Path(save_dir_path),
            k=k,
            bm25_dir_path=BM25_DIRPATH,
            chunks_file_path=CHUNK_FILEPATH,
            chroma_dir_path=CHROMA_DIRPATH,
            manifest_file_path=MANIFEST_FILEPATH,
            embedding_model_name=EMBEDDING_LLM_MODEL,
        )

    def answer(
        self,
        query: str,
        k: int = 10,
        details: bool = False,
        verbose: int = 0,
    ) -> None:
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)

        self._init_logging(verbose)

        entrypoint_answer(
            original_query=query,
            k=k,
            bm25_dir_path=BM25_DIRPATH,
            chunks_file_path=CHUNK_FILEPATH,
            chroma_dir_path=CHROMA_DIRPATH,
            manifest_file_path=MANIFEST_FILEPATH,
            embedding_model_name=EMBEDDING_LLM_MODEL,
            with_details=details,
        )

    def answer_dataset(
        self,
        save_dir_path: str = str(DATASET_ANSWER_OUTPUT),
        dataset_file_path: str = str(ANSWERED_FILEPATH),
        k: int = 10,
        verbose: int = 0,
    ) -> None:
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)

        self._init_logging(verbose)

        entrypoint_answer_dataset(
            dataset_file_path=Path(dataset_file_path),
            save_dir_path=Path(save_dir_path),
            k=k,
            bm25_dir_path=BM25_DIRPATH,
            chunks_file_path=CHUNK_FILEPATH,
            chroma_dir_path=CHROMA_DIRPATH,
            manifest_file_path=MANIFEST_FILEPATH,
            embedding_model_name=EMBEDDING_LLM_MODEL,
        )

    def evaluate(
        self,
        dataset_file_path: str = str(ANSWERED_FILEPATH),
        predictions_file_path: str = str(DATASET_SEARCH_OUTPUT),
        ks: str | tuple[int, ...] = (1, 3, 5, 10),
        verbose: int = 0,
    ) -> None:
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)

        self._init_logging(verbose)

        if isinstance(ks, str):
            ks = tuple(int(k.strip()) for k in ks.split(",") if k.strip())

        entrypoint_evaluate(
            dataset_file_path=Path(dataset_file_path),
            predictions_file_path=Path(predictions_file_path),
            ks=ks,
        )

    def _init_logging(self, verbose: int) -> None:
        LoggingSystem.global_setup(verbose)


def main() -> None:
    logger.info("Application starting...")
    os.environ.setdefault("PAGER", "cat")

    try:
        try:
            fire.Fire(App())
        except ValidationError as e:
            raise SchemaValidationError(e) from e
    except RagError as e:
        print(
            f"[{e.__class__.__name__}] {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except FireExit:
        sys.exit(2)
    except Exception as e:
        print(
            f"[{e.__class__.__name__}] An unexpected error occurred: {e}",
            file=sys.stderr,
        )
        sys.exit(3)


if __name__ == "__main__":
    main()
