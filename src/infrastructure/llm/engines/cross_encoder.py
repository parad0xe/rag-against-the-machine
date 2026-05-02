import logging
from typing import Any, Sequence, cast

import torch
from sentence_transformers import CrossEncoder
from transformers.utils import logging as transformers_logging

from src.application.ports.llm.engine import CrossEncoderEnginePort

cast(Any, transformers_logging).disable_progress_bar()

logger = logging.getLogger(__file__)


class CrossEncoderEngine(CrossEncoderEnginePort):
    """
    Engine that uses a Cross-Encoder model to score query-chunk pairs.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes the Cross-Encoder engine.

        Args:
            model_name: The Hugging Face model identifier.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CrossEncoder model {model_name}")

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

    def predict_scores(self, query: str, chunks: Sequence[str]) -> list[float]:
        """
        Predicts relevance scores for the given query and chunks.

        Args:
            query: The search query.
            chunks: A list of text chunks.

        Returns:
            A list of floats representing the relevance scores.
        """
        if not chunks:
            return []

        pairs = [[query, chunk] for chunk in chunks]

        with torch.inference_mode():
            scores = self.model.predict(pairs)

        return [float(score) for score in scores]
