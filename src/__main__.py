import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import fire
from fire.core import FireExit
from pydantic import (
    NonNegativeInt,
    TypeAdapter,
    ValidationError,
)

from src.commands.index import entrypoint_index
from src.domain.exceptions.base import RagError
from src.domain.exceptions.schema import SchemaValidationError
from src.logging import LoggingSystem

logger = logging.getLogger(__file__)

DEFAULT_INDEX_PATH: str = "vllm-0.10.1"

PROCESSED_DIR: Path = Path("data/processed")

BM25_DIRPATH: Path = PROCESSED_DIR / "bm25_index"
CHROMA_DIRPATH: Path = PROCESSED_DIR / "chroma_index"

STATS_FILEPATH: Path = PROCESSED_DIR / "stats.dat"
CHUNK_FILEPATH: Path = PROCESSED_DIR / "chunks.json"

LLM_MODEL: str = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # "all-MiniLM-L6-v2"
)


class App:
    def __init__(self) -> None:
        os.makedirs(PROCESSED_DIR, exist_ok=True)

    def index(
        self,
        path: str = DEFAULT_INDEX_PATH,
        extensions: str | tuple[str] = "*",
        chunk_size: int = 2000,
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
            extensions=extensions,
            chunk_size=chunk_size,
            bm25_dirpath=BM25_DIRPATH,
            chroma_dirpath=CHROMA_DIRPATH,
            embedding_model_name=LLM_MODEL,
            with_semantic=semantic,
        )
        # entrypoint_index(
        #    path=Path(path),
        #    extensions=extensions,
        #    chunk_size=chunk_size,
        #    bm25_dirpath=BM25_DIRPATH,
        #    chroma_dirpath=CHROMA_DIRPATH,
        #    stats_filepath=STATS_FILEPATH,
        #    chunks_filepath=CHUNK_FILEPATH,
        #    embedding_model_name=LLM_MODEL,
        #    semantic=semantic,
        # )

    def search(
        self,
        query: str,
        k: int = 10,
        verbose: int = 0,
    ) -> None:
        """
        Search for a single query

        Search this knowledge base to find relevant code
            snippets and documentation for given questions
        """
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)

        self._init_logging(verbose)

        # entrypoint_search(
        #    query=query,
        #    k=k,
        #    index_dirpath=BM25_DIRPATH,
        #    stats_filepath=STATS_FILEPATH,
        # )

    def search_dataset(self, verbose: int = 0) -> None:
        """
        Process multiple questions and output search results
        """
        self._init_logging(verbose)
        raise NotImplementedError("App.search_dataset")

    def answer(self, verbose: int = 0) -> None:
        """
        Answer a single question with context
        """
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)

        self._init_logging(verbose)
        raise NotImplementedError("App.answer")

    def answer_dataset(self, verbose: int = 0) -> None:
        """
        Answer a multiple questions with context
        """
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)

        self._init_logging(verbose)
        raise NotImplementedError("App.answer_dataset")

    def evaluate(self, verbose: int = 0) -> None:
        """
        Evaluate search results against ground truth
        """
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)

        self._init_logging(verbose)
        raise NotImplementedError("App.evaluate")

    def _init_logging(self, verbose: int) -> None:
        LoggingSystem.global_setup(verbose)


def main() -> None:
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
