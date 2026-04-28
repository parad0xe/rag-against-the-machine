import torch
from langdetect import DetectorFactory, LangDetectException, detect
from transformers import pipeline

from src.application.ports.translator import TranslatorInterface

DetectorFactory.seed = 0


class HuggingFaceTranslator(TranslatorInterface):
    def __init__(self, model: str = "Helsinki-NLP/opus-mt-mul-en") -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.translator = pipeline("translation", model=model, device=device)

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
            return detect(text) == "en"
        except LangDetectException:
            return True
