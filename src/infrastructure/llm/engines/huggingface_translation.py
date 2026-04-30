import logging
from typing import Any, cast

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from src.application.ports.llm.engine import TranslationEnginePort
from src.config import settings
from src.domain.exceptions.base import RagError

cast(Any, transformers_logging).disable_progress_bar()

logger = logging.getLogger(__file__)


class HuggingFaceTranslationEngine(TranslationEnginePort):
    def __init__(self, model_name: str = settings.translator_model) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model_name = model_name
        self.model: Any | None = None

    def translate(self, text: str) -> str:
        if self.model is None:
            logger.info(f"Loading translation model: {self._model_name}")
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._model_name
                ).to(self.device)

                if self.model:
                    self.model.eval()
            except MemoryError as e:
                raise RagError("Failed to load LLM into memory") from e

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = cast(Any, self.model).generate(
                **inputs,
                max_new_tokens=512,
            )

        translation = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        return str(translation)
