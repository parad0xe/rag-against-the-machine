import torch
from langdetect import DetectorFactory, LangDetectException, detect
from transformers import pipeline

from src.config import settings

DetectorFactory.seed = 0


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
            max_length=1024,
        )

        return str(result[0]["translation_text"])

    def _is_english(self, text: str) -> bool:
        try:
            return bool(detect(text) == "en")
        except LangDetectException:
            return True
