import logging
from pathlib import Path
from typing import Generator

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.infrastructure.index_stores.base import BaseIndexStoreSync

logger = logging.getLogger(__file__)


class ChromaIndexStoreSync(BaseIndexStoreSync):
    """
    Synchronization implementation for the Chroma vector index.
    """

    def __init__(
        self,
        dir_path: Path,
        embedding_model_name: str,
        batch_size: int = 32,
        addition_enable: bool = True,
    ) -> None:
        """
        Initializes the Chroma sync store.

        Args:
            dir_path: Path to the ChromaDB storage directory.
            embedding_model_name: Name of the SentenceTransformer model.
            batch_size: Number of chunks per embedding/upsert batch.
            addition_enable: Whether adding documents is allowed.
        """
        super().__init__(name="Chroma", addition_enable=addition_enable)
        self._dir_path = dir_path
        self._embedding_model_name = embedding_model_name
        self._batch_size = batch_size

    def commit(
        self, require_reset: bool = False
    ) -> Generator[tuple[int, int, str], None, None]:
        """
        Persists staged additions and deletions to ChromaDB.

        Args:
            require_reset: Whether to clear the collection before starting.

        Yields:
            Tuple of (current_step, total_steps, description).
        """
        client = chromadb.PersistentClient(
            path=str(self._dir_path),
            settings=Settings(anonymized_telemetry=False),
        )

        if require_reset:
            logger.info(f"[{self.__class__.__name__}] Resetting collection.")
            try:
                client.delete_collection(name="chunks")
            except Exception:
                pass

        collection = client.get_or_create_collection(name="chunks")

        if self._delete_chunk_ids:
            collection.delete(ids=list(self._delete_chunk_ids))
            logger.debug(
                f"[{self.__class__.__name__}] Removed "
                f"{len(self._delete_chunk_ids)} chunks from storage."
            )

        if self._add_documents:
            total_chunks = sum(len(d.chunks) for d in self._add_documents)
            batches = list(self._batches(self._batch_size))
            total_batches = len(batches)

            yield (
                0,
                total_batches,
                f"Loading model {self._embedding_model_name}",
            )
            model = SentenceTransformer(self._embedding_model_name)

            yield 0, total_batches, f"Preparing {total_chunks} chunks"

            for i, (batch_chunks, batch_ids) in enumerate(batches, 1):
                yield i, total_batches, f"Upserting batch {i}/{total_batches}"

                embeddings = model.encode(batch_chunks, convert_to_numpy=True)
                collection.upsert(
                    embeddings=embeddings.tolist(), ids=batch_ids
                )
        else:
            yield 1, 1, "No chunks to add"

        self._add_documents.clear()
        self._delete_chunk_ids.clear()

    def _batches(
        self, batch_size: int
    ) -> Generator[tuple[list[str], list[str]], None, None]:
        """
        Helper to group staged documents into fixed-size batches.

        Args:
            batch_size: Number of chunks per batch.

        Yields:
            Tuple of (batch_chunks, batch_ids).
        """
        batch_chunks = []
        batch_ids = []
        for doc in self._add_documents:
            for i, chunk in enumerate(doc.chunks):
                batch_chunks.append(chunk)
                batch_ids.append(doc.chunk_ids[i])
                if len(batch_chunks) >= batch_size:
                    yield batch_chunks, batch_ids
                    batch_chunks = []
                    batch_ids = []
        if batch_chunks:
            yield batch_chunks, batch_ids
