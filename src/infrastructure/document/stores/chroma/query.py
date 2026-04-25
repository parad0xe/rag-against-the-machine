import logging
from pathlib import Path

import chromadb
from chromadb import Collection
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.infrastructure.document.stores.base import IndexStoreQuery

logger = logging.getLogger(__file__)


class ChromaIndexStoreQuery(IndexStoreQuery):
    @property
    def name(self) -> str:
        return "Chroma"

    def __init__(
        self,
        dir_path: Path,
        embedding_model_name: str,
        enable: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__(dir_path, enable, weight)
        self._embedding_model_name: str = embedding_model_name
        self._collection: Collection | None = None
        self._model: SentenceTransformer | None = None

    def search(self, query: str, k: int) -> list[str] | None:
        if not self.enable or not self._dir_path.exists():
            return []

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
