import logging

from langdetect import DetectorFactory, LangDetectException, detect

from src.application.ports.llm import (
    LLMTranslatorPort,
    TranslationEnginePort,
)

DetectorFactory.seed = 0
logger = logging.getLogger(__file__)


class QueryTranslatorService(LLMTranslatorPort):
    def __init__(self, translation_engine: TranslationEnginePort) -> None:
        self._engine = translation_engine

    def translate_to_english(self, text: str) -> str:
        if not text.strip():
            return ""

        if self._is_english(text):
            return text

        try:
            translation = self._engine.translate(text)

            if not translation:
                logger.warning("Empty translation generated. Skipped.")
                return text

            logger.info(f"Translated text: '{translation}'")
            return translation

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
