import logging
from typing import Any

import torch
from langdetect import DetectorFactory, LangDetectException, detect
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from src.config import settings
from src.domain.exceptions.base import RagError

DetectorFactory.seed = 0
transformers_logging.disable_progress_bar()
logger = logging.getLogger(__file__)


class TextGenerationTranslatorLLM:
    def __init__(self, model: str = settings.translator_model) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._model_name = model
        self.model: Any | None = None

    def translate_to_english(self, text: str) -> str:
        if not text.strip():
            return ""

        if self._is_english(text):
            return text

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

        if not self.model:
            logger.error(
                f"Failed to load translation model: {self._model_name}. "
                "Skipped"
            )
            return text

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                )

            translation = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            if not translation:
                logger.warning("Empty translation generated. Skipped.")
                return text

            logger.info(f"Translate <{text}> -> <{translation}>")

            return str(translation)

        except Exception as e:
            logger.error(
                f"Translation failed ({e.__class__.__name__}). "
                "Fallback to original text."
            )
            return text

    def _is_english(self, text: str) -> bool:
        text_sample = text[:500]
        try:
            return bool(detect(text_sample) == "en")
        except LangDetectException:
            return True
