import glob
import json
import logging
import os
import sys

import bm25s
import fire
from langchain_core.documents import Document
from pydantic import (
    NonNegativeInt,
    PositiveInt,
    TypeAdapter,
    ValidationError,
)
from tqdm import tqdm

from src.chain import TextSplitter
from src.exceptions.base import RagError, SchemaValidationError
from src.logging import LoggingSystem
from src.utils import pluralize

logger = logging.getLogger(__file__)

PROJECT_ROOT = "vllm-0.10.1"


def get_filepaths() -> None:
    pass


def create_chunks() -> None:
    pass


def retrieve(query: str, k: PositiveInt) -> None:
    with open("data/processed/info.dat", "r") as f:
        info = json.load(f)

    print(info)
    reloaded_retriever = bm25s.BM25.load(
        "data/processed/bm25_index", load_corpus=True
    )

    # query = "does the fish purr like a cat?"
    query_tokens = bm25s.tokenize(query)

    results, scores = reloaded_retriever.retrieve(
        query_tokens,
        k=min(max(0, k), info.get("num_documents")),
        n_threads=8,
    )

    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(doc.get("filepath"), doc.get("index"), score)


class App:
    def index(
        self,
        path: str = "vllm-0.10.1",
        verbose: int = 0,
        extensions: tuple | str = ("py", "md"),
        chunk_size: int = 2000,
    ) -> None:
        path = TypeAdapter(str).validate_python(path)
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)
        chunk_size = TypeAdapter(PositiveInt).validate_python(chunk_size)
        extensions = TypeAdapter(tuple | str).validate_python(extensions)

        self._init_logging(verbose)
        # TD: se renseigner sur chromadb
        # TD: se renseigner sur langchain (smart chunking)

        if not os.path.exists(path):
            # TD: custom exception
            raise Exception("path not exists")

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

        # TD: creer des chunks de mes documents
        # TD: indexer les chunks
        # TD: sauvegarder les chunks et initialiser ma db ?

        os.makedirs("data/processed/", exist_ok=True)

        ll: list[Document] = []
        documents: list[str] = []
        metadata: list[dict[str, str | int]] = []
        for filepath in tqdm(filepaths):
            with open(f"{filepath}", "r") as f:
                document: str = f.read()

                ll.append(Document(page_content=document))

                splitter = TextSplitter.from_filename(
                    filepath,
                    chunk_size=chunk_size,
                    add_start_index=True,
                )
                chunks = splitter.split_text(document)
                # create hash
                # compare the hash
                # skip or continue
                for k, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append(
                        {"text": chunk, "filepath": filepath, "index": k}
                    )

        if len(documents) == 0:
            raise NotImplementedError("Empty document")

        with open("data/processed/info.dat", "w") as f:
            json.dump({"num_documents": len(documents)}, f)

        document_tokens = bm25s.tokenize(documents)

        retriever = bm25s.BM25(corpus=metadata)
        retriever.index(document_tokens)
        retriever.save("data/processed/bm25_index")

        raise NotImplementedError("App.index")

    def search(
        self,
        query: str,
        verbose: int = 0,
        k: int = 10,
    ) -> None:
        """
        Search for a single query

        Search this knowledge base to find relevant code
            snippets and documentation for given questions
        """
        query = TypeAdapter(str).validate_python(query)
        verbose = TypeAdapter(NonNegativeInt).validate_python(verbose)
        k = TypeAdapter(PositiveInt).validate_python(k)

        self._init_logging(verbose)
        retrieve(query, k)
        raise NotImplementedError("App.search")

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
        app = App()

        try:
            fire.Fire(
                {
                    "index": app.index,
                    "search": app.search,
                    "search_dataset": app.search_dataset,
                    "answer": app.answer,
                    "answer_dataset": app.answer_dataset,
                    "evaluate": app.evaluate,
                }
            )
        except ValidationError as e:
            raise SchemaValidationError(e) from e
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
