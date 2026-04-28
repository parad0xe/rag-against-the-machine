from abc import ABC, abstractmethod
from typing import Generator


class LLMAssistantInterface(ABC):
    @abstractmethod
    def generate_answer(
        self, query: str, context: str
    ) -> Generator[str, None, None]: ...


class LLMTranslatorInterface(ABC):
    @abstractmethod
    def translate_to_english(self, text: str) -> str: ...
