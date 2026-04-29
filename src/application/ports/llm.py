from typing import Generator, Protocol


class LLMAssistantPort(Protocol):
    def generate_answer(
        self, query: str, context: str
    ) -> Generator[str, None, None]: ...


class LLMTranslatorPort(Protocol):
    def translate_to_english(self, text: str) -> str: ...
