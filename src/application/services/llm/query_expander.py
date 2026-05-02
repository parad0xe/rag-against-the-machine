import logging
import textwrap

from src.application.ports.llm.engine import (
    ChatMessage,
    TextGenerationEnginePort,
)
from src.application.ports.llm.llm import LLMQueryExpanderPort

logger = logging.getLogger(__file__)


class QueryExpanderService(LLMQueryExpanderPort):
    """
    Service that uses an LLM to generate keywords for query expansion.
    """

    def __init__(self, llm_engine: TextGenerationEnginePort) -> None:
        """
        Initializes the query expander service.

        Args:
            llm_engine: Port for the text generation engine.
        """
        self._llm_engine = llm_engine

    def expand_query(self, query: str) -> str:
        """
        Generates expansion keywords for the provided query.

        Args:
            query: The original user query.

        Returns:
            A comma-separated string of expansion keywords.
        """
        system_prompt = textwrap.dedent(
            """
            You are a strict keyword extraction engine.
            Extract technical keywords, synonyms, and broader concepts
            from the user's query.
            Output ONLY a single line of comma-separated words.
            CRITICAL: NO markdown, NO bullet points, NO lists,
            NO conversational text.
            """
        ).strip()

        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "How does the app validate the authentication token?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "JWT, expiration, signature, AuthGuard, "
                    "middleware, bearer, payload, decode"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Why is my React component re-rendering infinitely?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "React, component, re-render, infinite loop, "
                    "useEffect, state dependency, hooks, render cycle"
                ),
            },
            {"role": "user", "content": query},
        ]

        try:
            result = self._llm_engine.generate(
                messages=messages,
                stream=False,
                max_new_tokens=150,
                do_sample=False,
                repetition_penalty=1.0,
                enable_thinking=False,
            )
            return str(result).strip()

        except Exception as e:
            logger.error(f"Query expansion failed ({e.__class__.__name__}).")
            return ""
