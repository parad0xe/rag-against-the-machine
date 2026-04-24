import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.domain.models.document import Document, DocumentStatus
from src.infrastructure.stores.base import BaseStore

logger = logging.getLogger(__file__)


class ChromaStore(BaseStore):
    @property
    def name(self) -> str:
        return "Chroma"

    def __init__(
        self,
        dirpath: Path,
        embedding_model_name: str,
        batch_size: int = 32,
        enable: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(dirpath, enable, weight)
        self._embedding_model_name: str = embedding_model_name
        self._batch_size: int = batch_size
        self._collection = None
        self._model = None

    def search(self, query: str, k: int) -> list[str]:
        if not self.enable or not self._dirpath.exists():
            return []

        import chromadb
        from chromadb.config import Settings

        if self._collection is None:
            client = chromadb.PersistentClient(
                path=str(self._dirpath),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(name="chunks")

        if self._model is None:
            self._model = SentenceTransformer(self._embedding_model_name)

        query_embedding = self._model.encode(query, convert_to_numpy=True)
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
        )

        if not results["ids"]:
            return []

        return results["ids"][0]

    def add(self, document: Document, status: DocumentStatus) -> None:
        if status == DocumentStatus.NOTHING_TO_DO:
            return
        super().add(document, status)

    def commit(self, require_reset_before: bool) -> None:
        if not self._add_documents and not self._delete_chunk_ids:
            return

        logger.info(
            f"[{self.__class__.__name__}] Synchronizing index: "
            f"{len(self._add_documents)} additions, "
            f"{len(self._delete_chunk_ids)} deletions."
        )

        import chromadb
        from chromadb.config import Settings

        self._dirpath.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(self._dirpath),
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )
        if require_reset_before:
            logger.info(f"[{self.__class__.__name__}] Resetting collection.")
            try:
                client.delete_collection(name="chunks")
            except ValueError as e:
                logger.warning(
                    f"[{self.__class__.__name__}] Skipped reset: {e}"
                )

        collection = client.get_or_create_collection(name="chunks")

        if self._delete_chunk_ids:
            collection.delete(ids=list(self._delete_chunk_ids))
            logger.debug(
                f"[{self.__class__.__name__}] Removed "
                f"{len(self._delete_chunk_ids)} chunks from storage."
            )

        if self._add_documents:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"[{self.__class__.__name__}] Loading embedding model: "
                f"'{self._embedding_model_name}'"
            )
            model = SentenceTransformer(self._embedding_model_name)

            chunks: list[str] = []
            chunk_ids: list[str] = []
            for doc in self._add_documents:
                chunks.extend(doc.chunks)
                chunk_ids.extend(doc.chunk_ids)

            logger.info(
                f"[{self.__class__.__name__}] Encoding {len(chunks)} chunks "
                f"in batches of {self._batch_size}."
            )
            for i in tqdm(
                range(0, len(chunks), self._batch_size),
                desc="Store chunk embeddings",
            ):
                batch_chunks = chunks[i : i + self._batch_size]
                batch_ids = chunk_ids[i : i + self._batch_size]

                embeddings = model.encode(
                    batch_chunks,
                    convert_to_numpy=True,
                )
                collection.upsert(
                    embeddings=embeddings.tolist(), ids=batch_ids
                )

        logger.info(f"[{self.__class__.__name__}] Synchronization complete.")
        self._clear_state()
