import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.infrastructure.index_stores.base import BaseIndexStoreQuery

logger = logging.getLogger(__file__)


class ChromaIndexStoreQuery(BaseIndexStoreQuery):
    def __init__(
        self,
        dir_path: Path,
        embedding_model_name: str,
        enable: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name="Chroma", enable=enable, weight=weight)
        self._dir_path = dir_path
        self._embedding_model_name = embedding_model_name
        self._collection: chromadb.Collection | None = None
        self._model: SentenceTransformer | None = None

    def search(self, query: str, k: int) -> list[str] | None:
        if self._collection is None:
            client = chromadb.PersistentClient(
                path=str(self._dir_path),
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

        return results["ids"][0] if results["ids"] else []
