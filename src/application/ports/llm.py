from typing import Generator, Protocol, Sequence, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class TextGenerationEnginePort(Protocol):
    def generate(
        self, messages: list[ChatMessage], stream: bool = False, **kwargs
    ) -> str | Generator[str, None, None]: ...


class TranslationEnginePort(Protocol):
    def translate(self, text: str) -> str: ...


class ReRankerEnginePort(Protocol):
    def predict_scores(
        self, query: str, chunks: Sequence[str]
    ) -> list[float]: ...


class LLMQueryExpanderPort(Protocol):
    def expand_query(self, query: str) -> str: ...


class LLMAssistantPort(Protocol):
    def generate_answer(
        self, query: str, context: str
    ) -> Generator[str, None, None]: ...


class LLMTranslatorPort(Protocol):
    def translate_to_english(self, text: str) -> str: ...


class LLMReRankerPort(Protocol):
    def rerank(
        self, query: str, chunks: Sequence[str], top_k: int = 5
    ) -> list[str]: ...
