import logging

import torch
from langdetect import DetectorFactory, LangDetectException, detect
from transformers import pipeline

from src.config import settings

DetectorFactory.seed = 0

logger = logging.getLogger(__file__)


class TextGenerationTranslatorLLM:
    def __init__(self, model: str = settings.translator_model) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.translator = pipeline(
            "text-generation", model=model, device=device
        )

    def translate_to_english(self, text: str) -> str:
        if not text.strip():
            return ""

        if self._is_english(text):
            return text

        result = self.translator(
            text,
            max_length=512,
            truncation=True,
        )

        translation_dict = result[0]
        translation = translation_dict.get(
            "translation_text"
        ) or translation_dict.get("generated_text")

        if not translation:
            logger.error("Failed to translate text. Skipped")
            return text
        return translation

    def _is_english(self, text: str) -> bool:
        try:
            return bool(detect(text) == "en")
        except LangDetectException:
            return True
