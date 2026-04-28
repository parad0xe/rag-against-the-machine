import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm.rich import tqdm

from src.application.ports.index_store.store import IndexStoreSyncInterface

logger = logging.getLogger(__file__)


class ChromaIndexStoreSync(IndexStoreSyncInterface):
    @property
    def name(self) -> str:
        return "Chroma"

    def __init__(
        self,
        dir_path: Path,
        embedding_model_name: str,
        batch_size: int = 32,
        enable: bool = True,
        addition_enable: bool = True,
    ) -> None:
        super().__init__(dir_path, enable, addition_enable)
        self._embedding_model_name: str = embedding_model_name
        self._batch_size: int = batch_size

    def _perform_commit(self, require_reset: bool) -> None:
        logger.info(
            f"[{self.__class__.__name__}] Synchronizing index: "
            f"{len(self._add_documents)} additions, "
            f"{len(self._delete_chunk_ids)} deletions."
        )

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
            logger.info(
                f"[{self.__class__.__name__}] Loading embedding model: "
                f"'{self._embedding_model_name}'"
            )
            model = SentenceTransformer(self._embedding_model_name)

            total_chunks = sum(len(d.chunks) for d in self._add_documents)
            logger.info(
                f"[{self.__class__.__name__}] Encoding {total_chunks} chunks "
                f"in batches of {self._batch_size}."
            )
            with tqdm(
                total=total_chunks, desc="Store chunk embeddings"
            ) as pbar:
                for batch_chunks, batch_ids in self._batches(self._batch_size):
                    embeddings = model.encode(
                        batch_chunks, convert_to_numpy=True
                    )
                    collection.upsert(
                        embeddings=embeddings.tolist(), ids=batch_ids
                    )
                    pbar.update(len(batch_chunks))

        logger.info(f"[{self.__class__.__name__}] Synchronization complete.")

        self._clear_state()
