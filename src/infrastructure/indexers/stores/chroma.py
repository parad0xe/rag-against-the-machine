import logging
from pathlib import Path

from tqdm import tqdm

from src.domain.models.document import Document, DocumentStatus
from src.infrastructure.indexers.stores.base import BaseIndexStore

logger = logging.getLogger(__file__)


class ChromaIndexStore(BaseIndexStore):
    def __init__(
        self,
        dirpath: Path,
        embedding_model_name: str,
        batch_size: int = 32,
        enable: bool = True,
    ) -> None:
        super().__init__(dirpath, enable)
        self._embedding_model_name: str = embedding_model_name
        self._batch_size: int = batch_size

    def add(self, document: Document, status: DocumentStatus) -> None:
        if status == DocumentStatus.NOTHING_TO_DO:
            return
        super().add(document, status)

    def commit(self, require_reset_before: bool) -> None:
        if not self._enable:
            return

        if len(self._add_documents) == 0 and len(self._delete_chunk_ids) == 0:
            return

        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(self._dirpath),
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )
        if require_reset_before:
            client.delete_collection(name="chunks")
        collection = client.get_or_create_collection(name="chunks")

        if len(self._delete_chunk_ids) > 0:
            collection.delete(ids=list(self._delete_chunk_ids))
            logger.info(f"{len(self._delete_chunk_ids)} deleted chunks.")

        if len(self._add_documents) > 0:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._embedding_model_name)
            chunks: list[str] = []
            chunk_ids: list[str] = []
            for doc in self._add_documents:
                chunks.extend(doc.chunks)
                chunk_ids.extend(doc.chunk_ids)

            for i in tqdm(
                range(0, len(chunks), self._batch_size),
                desc="Convert chunks to embeddings",
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
