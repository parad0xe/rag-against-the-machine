import contextlib
import glob
import logging
import os
import sys

import bm25s
import fire
from tqdm import tqdm

from src.exceptions.base import RagError
from src.logging import LoggingSystem

logger = logging.getLogger(__file__)


PROJECT_ROOT = "vllm-0.10.1"


def pluralize(word: str, count: int) -> str:
    if count <= 1:
        return word
    return word + "s"


class App:
    def __init__(
        self, verbose: int = 0, chunk_size: int = 2000, k: int = 10
    ) -> None:
        self._verbose: int = verbose
        self._chunk_size: int = chunk_size
        self._k: int = k
        LoggingSystem.global_setup(verbose)

    def index(self, extensions: tuple = ("py", "md")) -> None:
        """
        Index the repository

        Ingest the vLLM repository (provided as attachment)
        and create a searchable knowledge base
        """
        if isinstance(extensions, str):
            exts = [extensions]
        else:
            exts = list(extensions)
        logger.info(f"load extensions: {exts}")
        filepaths: list[str] = []
        for ext in exts:
            filepaths += glob.glob(
                f"{PROJECT_ROOT}/**/*.{ext}", recursive=True
            )
        logger.info(
            f"{len(filepaths)} {pluralize('file', len(filepaths))} "
            f"founded for extensions {exts}"
        )

        os.makedirs("data/processed/", exist_ok=True)

        for filepath in tqdm(filepaths):
            with open(f"{filepath}", "r") as f:
                document = f.readlines()
                if not document:
                    continue

                document_tokens = bm25s.tokenize(document, show_progress=False)

                retriever = bm25s.BM25()
                retriever.index(document_tokens, show_progress=False)
                with (
                    open(os.devnull, "w") as fnull,
                    contextlib.redirect_stderr(fnull),
                ):
                    retriever.save(
                        f"data/processed/{filepath}_index_bm25",
                        corpus=document,
                    )

                # query = "does the fish purr like a cat?"
                # query_tokens = bm25s.tokenize(query)

                # results, scores = retriever.retrieve(
                #    query_tokens,
                #    k=min(len(document), max(0, self._k)),
                #    n_threads=8,
                #    chunksize=self._chunk_size,
                # )

                # for i in range(results.shape[1]):
                #    doc, score = results[0, i], scores[0, i]

        raise NotImplementedError("App.index")

    def search(self) -> None:
        """
        Search for a single query

        Search this knowledge base to find relevant code
            snippets and documentation for given questions
        """
        raise NotImplementedError("App.search")

    def search_dataset(self) -> None:
        """
        Process multiple questions and output search results

        Search this knowledge base to find relevant code
            snippets and documentation for given questions
        """
        raise NotImplementedError("App.search_dataset")

    def answer(self) -> None:
        """
        Answer a single question with context

        Answer questions using an LLM (Qwen/Qwen3-0.6B) with
            the retrieved context
        """
        raise NotImplementedError("App.answer")

    def answer_dataset(self) -> None:
        """
        Answer a multiple questions with context

        Answer questions using an LLM (Qwen/Qwen3-0.6B) with
            the retrieved context
        """
        raise NotImplementedError("App.answer_dataset")

    def evaluate(self) -> None:
        """
        Evaluate search results against ground truth

        Evaluate your retrieval system’s quality using recall@k metrics
        """
        raise NotImplementedError("App.evaluate")


def main() -> None:
    try:
        fire.Fire(App)
    except RagError as e:
        print(
            f"[{e.__class__.__name__}] {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except BaseException as e:
        print(
            f"[{e.__class__.__name__}] An unexpected error occurred: {e}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
