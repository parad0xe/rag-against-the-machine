import logging
from typing import Sequence

from src.application.ports.llm.engine import CrossEncoderEnginePort
from src.application.ports.llm.llm import LLMReRankerPort

logger = logging.getLogger(__file__)


class RerankerService(LLMReRankerPort):
    """
    Service that uses a cross-encoder to improve retrieval relevance.
    """

    def __init__(self, cross_encoder_engine: CrossEncoderEnginePort) -> None:
        """
        Initializes the reranker service.

        Args:
            cross_encoder_engine: Port for the cross-encoder engine.
        """
        self._cross_encoder_engine = cross_encoder_engine

    def rerank(
        self, query: str, chunks: Sequence[str], top_k: int = 5
    ) -> list[str]:
        """
        Re-ranks a list of chunks based on their relevance to the query.

        Args:
            query: The search query.
            chunks: The list of retrieved chunks to re-rank.
            top_k: The number of top chunks to return.

        Returns:
            The top_k most relevant chunks.
        """
        if not chunks:
            return []

        try:
            scores = self._cross_encoder_engine.predict_scores(query, chunks)
        except Exception as e:
            logger.error(
                f"Re-ranking failed ({e.__class__.__name__}). "
                "Fallback to original order."
            )
            return list(chunks)[:top_k]

        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in scored_chunks[:top_k]]
