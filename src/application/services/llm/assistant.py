import logging
import textwrap
from typing import Generator

from src.application.ports.llm.engine import (
    ChatMessage,
    TextGenerationEnginePort,
)
from src.application.ports.llm.llm import LLMAssistantPort

logger = logging.getLogger(__file__)


class AssistantService(LLMAssistantPort):
    def __init__(self, llm_engine: TextGenerationEnginePort) -> None:
        self._llm_engine = llm_engine

    def generate_answer(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        if not context.strip():
            yield "I could not find enough information to answer."
            return

        system_prompt = textwrap.dedent(
            """
            You are a strict, robotic code analysis AI. You must answer
            the user's question using ONLY the information
            provided in the context.

            If the answer cannot be found, reply ONLY with:
            "I could not find enough information to answer."

            You are FORBIDDEN to use conversational filler
            (e.g., "Here is the answer", "Based on the context").

            Sourcing all the best source files.

            You MUST use this EXACT format:
            ### Answer
            [Your concise answer here]

            ### Sources
            - [File: <file_path> (Chars: <start>-<end>)]

            EXAMPLE OF VALID RESPONSE:
            ### Answer
            The `AuthenticationService` uses JWT
            tokens with a 15-minute expiration time.

            ### Sources
            - [File: src/auth/service.py (Chars: 450-512)]
            """
        ).strip()

        assistant_prompt = textwrap.dedent(
            f"""
            <context>
            {context}
            </context>
            """
        )

        user_prompt = textwrap.dedent(
            f"""
            Question: {query}
            """
        ).strip()

        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": user_prompt},
        ]

        stream = self._llm_engine.generate(
            messages=messages,
            stream=True,
            max_new_tokens=2048,
            repetition_penalty=1.0,
            do_sample=False,
            enable_thinking=True,
        )

        for token in stream:
            yield str(token)
