from typing import Any, Generator, Protocol, Sequence, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class TextGenerationEnginePort(Protocol):
    def generate(
        self,
        messages: list[ChatMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Generator[str, None, None]: ...


class TranslationEnginePort(Protocol):
    def translate(self, text: str) -> str: ...


class CrossEncoderEnginePort(Protocol):
    def predict_scores(
        self, query: str, chunks: Sequence[str]
    ) -> list[float]: ...
