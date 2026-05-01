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
        self, query: str, context: str, thinking: bool = False
    ) -> Generator[str, None, None]:
        if not context.strip():
            yield "I could not find enough information to answer."
            return

        logger.debug(f"Context length provided: {len(context)} characters.")

        system_prompt = textwrap.dedent(
            """
            You are a strict factual extraction model for RAG.

            Use only CONTEXT_START to CONTEXT_END.
            Do not use any outside knowledge.

            Follow these rules:
            - Answer only what is explicitly supported by the context.
            - Quote the exact evidence used in the context.
            - Do not add disclaimers, or conversational text.
            - Do not invent sources, citations, file path, or character ranges.

            If the answer is not clearly present in the context,
            respond exactly with:
            I could not find enough information to answer.

            However, if the answer is present, use exactly this format:

            ### Answer
            [factual answer]

            ### Sources
            - [File: <file_path> (Chars: <start>-<end>)]
                - [exact quote here]

            Output nothing else.
            """
        ).strip()

        user_prompt = textwrap.dedent(
            f"""
            CONTEXT_START
            {context}
            CONTEXT_END

            Question: {query}
            """
        ).strip()

        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug("Starting llm generation stream...")

        stream = self._llm_engine.generate(
            messages=messages,
            stream=True,
            max_new_tokens=2048,
            repetition_penalty=1.0,
            do_sample=False,
            enable_thinking=thinking,
        )

        token_count = 0
        for token in stream:
            token_count += 1
            yield str(token)

        logger.debug(
            f"LLM generation stream completed. streamed {token_count} tokens."
        )
