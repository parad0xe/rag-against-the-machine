from typing import Generator, Protocol, Sequence


class LLMQueryExpanderPort(Protocol):
    def expand_query(self, query: str) -> str: ...


class LLMAssistantPort(Protocol):
    def generate_answer(
        self, query: str, context: str, thinking: bool = False
    ) -> Generator[str, None, None]: ...


class LLMTranslatorPort(Protocol):
    def translate_to_english(self, text: str) -> str: ...


class LLMReRankerPort(Protocol):
    def rerank(
        self, query: str, chunks: Sequence[str], top_k: int = 5
    ) -> list[str]: ...
