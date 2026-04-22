import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

import bm25s
import chromadb
from chromadb.config import Settings
from pydantic import PositiveInt, validate_call
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.chain import CustomTextSplitter
from src.exceptions.document import NoDocumentError
from src.exceptions.schema import (
    SchemaInvalidJSONFormatError,
    SchemaInvalidJSONRootError,
)
from src.exceptions.storage import StorageDirNotFoundError
from src.models.chunk import ChunkMetadata
from src.models.manifest import CachedFile, Manifest
from src.models.stats import Stats
from src.utils.common_util import file_md5sum, md5sum, pluralize
from src.utils.path_util import get_extension

logger = logging.getLogger(__file__)


MAX_BATCH_SIZE = 32


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
    bm25_dirpath: Path = Path("data/processed/bm25_index"),
    chroma_dirpath: Path = Path("data/processed/chroma_index"),
    stats_filepath: Path = Path("data/processed/stats.dat"),
    chunks_filepath: Path = Path("data/processed/chunks.json"),
    manifest_filepath: Path = Path("data/processed/manifest.json"),
    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    semantic: bool = False,
) -> None:
    if not os.path.exists(path):
        raise StorageDirNotFoundError(path)

    exts: list[str] = extensions.replace(" ", "").replace(".", "").split(",")
    exts = ["*" if "*" in ext else ext for ext in exts if ext]
    if "*" in exts:
        exts = ["*"]

    stats = Stats()
    chunks: list[str] = []
    chunks_for_chroma: list[str] = []
    chunk_indexes = {"bm25": [], "chroma": {"delete": [], "new": []}}
    chunk_metadatas: dict[str, dict[str, Any]] = {}

    try:
        if chunks_filepath.exists():
            with open(chunks_filepath, "r") as f:
                chunk_metadatas = json.load(f)

            if not isinstance(chunk_metadatas, dict):
                raise SchemaInvalidJSONRootError(
                    expected=dict, context=chunks_filepath
                )
    except json.JSONDecodeError as e:
        raise SchemaInvalidJSONFormatError(
            context=chunks_filepath,
            lineno=e.lineno,
        ) from e

    manifest: Manifest = Manifest.load_from_file(
        manifest_filepath, embedding_model_name, chunk_size
    )
    expired_chunk_ids = manifest.sync(exts)
    chunk_indexes["chroma"]["delete"] += expired_chunk_ids

    filepaths: list[str] = __get_filepaths(path, exts, recursive=True)
    for filepath in tqdm(filepaths, desc="Processing documents"):
        file_id: str = md5sum(filepath)
        file_hash: str = file_md5sum(filepath)
        file_ext: str = get_extension(filepath)
        file_chunk_ids: list[str] = []

        try:
            with open(filepath, "r") as f:
                document: str = f.read()
        except UnicodeDecodeError:
            continue

        file_chunks = CustomTextSplitter.from_filename(
            filepath,
            chunk_size=chunk_size,
            add_start_index=True,
        ).split_text(document)

        cached_ext: dict = (
            manifest.files_by_ext[file_ext]
            if file_ext in manifest.files_by_ext
            else {}
        )
        cached_file: CachedFile | None = (
            cached_ext[file_id] if file_id in cached_ext else None
        )
        file_require_update = True
        if cached_file:
            if cached_file.filehash != file_hash:
                chunk_indexes["chroma"]["delete"] += cached_file.chunk_ids
                file_require_update = True
            else:
                file_require_update = False

        start_index = 0
        for chunk in file_chunks:
            chunk_length = len(chunk)
            chunk_start_index: int = start_index
            chunk_end_index: int = start_index + chunk_length
            chunk_id: str = (
                f"chunk_{file_id}_{chunk_start_index}_{chunk_end_index}"
            )
            chunk_hash: str = md5sum(chunk)

            file_chunk_ids.append(chunk_id)
            chunks.append(chunk_id)
            chunk_metadatas[chunk_id] = ChunkMetadata(
                **{
                    "text": chunk,
                    "hash": chunk_hash,
                    "file_path": filepath,
                    "first_character_index": chunk_start_index,
                    "last_character_index": chunk_end_index,
                }
            ).model_dump()
            start_index += chunk_length

        if not cached_file:
            manifest.files_by_ext.setdefault(file_ext, {})

        manifest.files_by_ext[file_ext][file_id] = CachedFile(
            **{
                "filepath": filepath,
                "filehash": file_hash,
                "chunk_ids": file_chunk_ids,
            }
        )
        chunk_indexes["bm25"].append(
            {"id": chunk_id for chunk_id in file_chunk_ids}
        )
        if (file_require_update and semantic) or not chroma_dirpath.exists():
            chunk_indexes["chroma"]["new"] += file_chunk_ids
            chunks_for_chroma += file_chunks

        stats.num_documents += 1
        stats.num_document_by_exts[file_ext] = (
            stats.num_document_by_exts.get(file_ext, 0) + 1
        )

    if not chunks:
        raise NoDocumentError(path)

    stats.num_chunks = len(chunks)
    stats.save(stats_filepath)

    logger.info(f"Pre-indexing stats: {stats}")
    logger.info(stats)

    chunk_tokens = bm25s.tokenize(chunks)
    retriever = bm25s.BM25(corpus=chunk_indexes["bm25"])
    retriever.index(chunk_tokens)
    retriever.save(bm25_dirpath)

    chroma_indexes = chunk_indexes["chroma"]["new"]
    chroma_indexes_delete = chunk_indexes["chroma"]["delete"]
    if len(chroma_indexes) > 0 and semantic:
        client = chromadb.PersistentClient(
            path=str(chroma_dirpath),
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        model = SentenceTransformer(embedding_model_name)
        collection = client.get_or_create_collection(name="chunks")

        logger.debug("Embedding model device: %s", model.device)

        if expired_chunk_ids:
            collection.delete(ids=chunk_indexes["chroma"]["delete"])
            logger.info(f"{len(expired_chunk_ids)} deleted chunks.")

        for i in tqdm(range(0, len(chunks_for_chroma), MAX_BATCH_SIZE)):
            batch_chunks = chunks_for_chroma[i : i + MAX_BATCH_SIZE]
            batch_ids = chroma_indexes[i : i + MAX_BATCH_SIZE]

            embeddings = model.encode(
                batch_chunks,
                convert_to_numpy=True,
            )
            collection.upsert(embeddings=embeddings.tolist(), ids=batch_ids)
    elif len(chroma_indexes_delete) and semantic:
        client = chromadb.PersistentClient(
            path=str(chroma_dirpath),
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )
        collection = client.get_or_create_collection(name="chunks")

        if expired_chunk_ids:
            collection.delete(ids=chunk_indexes["chroma"]["delete"])
            logger.info(f"{len(expired_chunk_ids)} deleted chunks.")

    with open(manifest_filepath, "w") as f:
        json.dump(manifest.model_dump(mode="json"), f)

    with open(chunks_filepath, "w") as f:
        json.dump(chunk_metadatas, f)

    print("DELETE:", len(chunk_indexes["chroma"]["delete"]))
    print("NEW:", len(chunk_indexes["chroma"]["new"]))

    logger.info("Indexing complete.")
