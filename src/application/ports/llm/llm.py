from typing import Generator, Protocol, Sequence


class LLMQueryExpanderPort(Protocol):
    """
    Port for expanding search queries with keywords and synonyms.
    """

    def expand_query(self, query: str) -> str:
        """
        Expand the input query into a set of technical keywords.

        Args:
            query: The original user query.

        Returns:
            A string containing comma-separated keywords.
        """
        ...


class LLMAssistantPort(Protocol):
    """
    Port for generating answers based on retrieved context.
    """

    def generate_answer(
        self, query: str, context: str, thinking: bool = False
    ) -> Generator[str, None, None]:
        """
        Generate a factual answer using the provided context.

        Args:
            query: The question to answer.
            context: The text content to use as reference.
            thinking: Whether to enable and show the model's "thinking"
                process.

        Yields:
            A single string token of the generated response.
        """
        ...


class LLMTranslatorPort(Protocol):
    """
    Port for translating various languages into English for the RAG system.
    """

    def translate_to_english(self, text: str) -> str:
        """
        Detect and translate the input text to English if necessary.

        Args:
            text: The source text.

        Returns:
            The English translation of the text.
        """
        ...


class LLMReRankerPort(Protocol):
    """
    Port for re-ranking retrieved chunks using a cross-encoder.
    """

    def rerank(
        self, query: str, chunks: Sequence[str], top_k: int = 5
    ) -> list[str]:
        """
        Re-order chunks based on their relevance to the query.

        Args:
            query: The search query.
            chunks: The list of text chunks to re-rank.
            top_k: The number of top-ranked chunks to return.

        Returns:
            A list of the top_k most relevant chunks.
        """
        ...
