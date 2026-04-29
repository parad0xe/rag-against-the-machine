import logging
import textwrap
from threading import Thread
from typing import Any, Generator, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from transformers.utils import logging as transformers_logging

from src.config import settings
from src.domain.exceptions.base import RagError

transformers_logging.disable_progress_bar()

logger = logging.getLogger(__file__)


class QwenAssistantLLM:
    def __init__(self, model_name: str = settings.llm_model) -> None:
        logger.info(f"Loading LLM {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self.model.eval()
        except MemoryError as e:
            raise RagError("Failed to load LLM into memory") from e

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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=8192,
            truncation=True,
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=2048,
            repetition_penalty=1.0,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        def generate_task() -> None:
            try:
                with torch.inference_mode():
                    cast(Any, self.model).generate(**generation_kwargs)
            except Exception as e:
                logger.error(
                    f"LLM generation failed ({e.__class__.__name__}): {e}"
                )

        thread = Thread(target=generate_task)
        thread.start()

        for new_text in streamer:
            yield new_text

    def expand_query(self, query: str) -> str:
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

        messages = [
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

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        try:
            with torch.inference_mode():
                outputs = cast(Any, self.model).generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]

            hypothetical_doc = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            return str(hypothetical_doc).strip()

        except Exception as e:
            logger.error(f"HyDE expansion failed ({e.__class__.__name__}).")
            return ""
