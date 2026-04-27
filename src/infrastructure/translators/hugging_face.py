from transformers import pipeline

from src.application.ports.translator import TranslatorInterface


class HuggingFaceTranslator(TranslatorInterface):
    def __init__(self, model: str = "Helsinki-NLP/opus-mt-mul-en") -> None:
        self.translator = pipeline(
            "translation",
            model=model,
        )

    def translate_to_english(self, text: str) -> str:
        if not text.strip():
            return ""

        result = self.translator(
            text,
            max_length=512,
        )

        return str(result[0]["translation_text"])
