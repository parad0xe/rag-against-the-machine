from typing import Any, Generator, Protocol, Sequence, TypedDict


class ChatMessage(TypedDict):
    """
    Representation of a message in a chat conversation.

    Attributes:
        role: The role of the message sender (e.g., 'system', 'user').
        content: The text content of the message.
    """

    role: str
    content: str


class TextGenerationEnginePort(Protocol):
    """
    Port for generating text using a Causal LLM.
    """

    def generate(
        self,
        messages: list[ChatMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Generator[str, None, None]:
        """
        Generate a response based on a list of chat messages.

        Args:
            messages: The conversation history.
            stream: Whether to stream the response tokens.
            **kwargs: Additional engine-specific generation parameters.

        Returns:
            The generated text string or a generator for streaming.
        """
        ...


class TranslationEnginePort(Protocol):
    """
    Port for translating text between languages.
    """

    def translate(self, text: str) -> str:
        """
        Translate the given text.

        Args:
            text: The source text to translate.

        Returns:
            The translated text.
        """
        ...


class CrossEncoderEnginePort(Protocol):
    """
    Port for scoring text pairs using a Cross-Encoder model.
    """

    def predict_scores(self, query: str, chunks: Sequence[str]) -> list[float]:
        """
        Predict relevance scores for a query and a set of chunks.

        Args:
            query: The search query.
            chunks: A sequence of text chunks to score.

        Returns:
            A list of floats representing the relevance scores.
        """
        ...
