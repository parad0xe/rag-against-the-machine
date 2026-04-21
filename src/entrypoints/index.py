import glob
import logging
import os
from pathlib import Path
from typing import Any

import bm25s
from pydantic import PositiveInt, validate_call
from tqdm import tqdm

from src.chain import CustomTextSplitter
from src.exceptions.document import NoDocumentError
from src.exceptions.storage import StorageDirNotFoundError
from src.models.chunk import ChunkMetadata
from src.models.stats import Stats
from src.utils.common_util import pluralize
from src.utils.path_util import get_extension

logger = logging.getLogger(__file__)


def __get_filepaths(
    basepath: Path,
    extensions: list[str],
    recursive: bool = False,
) -> list[str]:
    logger.info(f"Loading extensions: {extensions}")

    filepaths: list[str] = []
    for ext in extensions:
        path = basepath
        if recursive:
            path = path / "**"
        path = path / f"*.{ext}"
        filepaths += glob.glob(str(path), recursive=recursive)

    logger.info(
        f"{len(filepaths)} {pluralize('file', len(filepaths))} "
        f"found for extensions {extensions}"
    )

    return filepaths


@validate_call()
def entrypoint_index(
    path: Path = Path("vllm-0.10.1"),
    extensions: str = "*",
    chunk_size: PositiveInt = 2000,
    index_dirpath: Path = Path("data/processed/bm25_index"),
    stats_filepath: Path = Path("data/processed/stats.dat"),
) -> None:
    # TD: se renseigner sur chromadb
    # TD: se renseigner sur langchain (smart chunking)

    if not os.path.exists(path):
        raise StorageDirNotFoundError(path)

    exts: list[str] = extensions.replace(" ", "").replace(".", "").split(",")
    exts = ["*" if "*" in ext else ext for ext in exts if ext]
    if "*" in exts:
        exts = ["*"]
    # 0. [ok] discover useful repository files
    # 1. [ok] load documents with langchain loaders
    # 2. [ok] split documents with a strategy depending on file type
    #    - python code splitter
    #    - markdown/text splitter
    # 3. [ok] convert split documents into internal chunks with:
    #    file_path, first_character_index, last_character_index, content
    # 4. build a BM25 index with bm25s from chunk contents
    # 5. save chunks and bm25 index under data/processed/
    stats = Stats()
    chunks: list[str] = []
    metadata: list[dict[str, Any]] = []

    filepaths: list[str] = __get_filepaths(path, exts, recursive=True)
    for filepath in tqdm(filepaths, desc="Processing documents"):
        try:
            with open(filepath, "r") as f:
                document: str = f.read()
        except UnicodeDecodeError:
            continue

        document_chunks = CustomTextSplitter.from_filename(
            filepath,
            chunk_size=chunk_size,
            add_start_index=True,
        ).split_text(document)

        start_index = 0
        for chunk in document_chunks:
            chunk_length = len(chunk)
            chunks.append(chunk)
            metadata.append(
                ChunkMetadata(
                    **{
                        "text": chunk,
                        "file_path": filepath,
                        "first_character_index": start_index,
                        "last_character_index": start_index + chunk_length,
                    }
                ).model_dump()
            )
            start_index += chunk_length

        ext = get_extension(filepath)

        stats.num_documents += 1
        stats.num_document_by_exts[ext] = (
            stats.num_document_by_exts.get(ext, 0) + 1
        )

    if not chunks:
        raise NoDocumentError(path)

    stats.num_chunks = len(chunks)
    logger.info(f"Pre-indexing stats: {stats}")

    chunk_tokens = bm25s.tokenize(chunks)
    retriever = bm25s.BM25(corpus=metadata)
    retriever.index(chunk_tokens)
    retriever.save(index_dirpath)
    stats.save(stats_filepath)

    logger.info("Indexing complete.")
