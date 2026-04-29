import logging
from typing import Any, cast

import torch
from sentence_transformers import CrossEncoder
from transformers.utils import logging as transformers_logging

transformers_logging.disable_progress_bar()

logger = logging.getLogger(__file__)


class CrossEncoderReRanker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Cross-Encoder ReRanker {model_name}...")

        model_kwargs = (
            {"torch_dtype": cast(Any, torch).float16}
            if self.device == "cuda"
            else {}
        )

        self.model = CrossEncoder(
            model_name,
            device=self.device,
            max_length=512,
            model_kwargs=model_kwargs,
        )

    def rerank(
        self, query: str, chunks: list[str], top_k: int = 5
    ) -> list[str]:
        if not chunks:
            return []

        pairs = [[query, chunk] for chunk in chunks]

        try:
            with torch.inference_mode():
                scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(
                f"Re-ranking failed ({e.__class__.__name__}). "
                "Fallback to original order."
            )
            return chunks[:top_k]

        scored_chunks = list(zip(scores, chunks))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in scored_chunks[:top_k]]
