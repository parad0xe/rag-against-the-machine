import logging
import textwrap
from threading import Thread
from typing import Generator

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from src.application.ports.llm import LLMAssistantInterface
from src.domain.exceptions.base import RagError

logger = logging.getLogger(__file__)


class QwenAssistantLLM(LLMAssistantInterface):
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        logger.info(f"Loading LLM {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
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
            You are a strict code analysis AI. You must answer the user's
            question using ONLY the information provided in the context.
            If the answer cannot be found in the context, you must reply
            with exactly one sentence stating this, and you must NOT
            generate the "### Sources" section. Do not guess.
            """
        ).strip()

        user_prompt = textwrap.dedent(
            f"""
            <context>
            {context}
            </context>

            Question: {query}

            Provide your response using EXACTLY the following structure.
            Do not add any conversational filler.

            ### Answer
            [Write your concise answer here, based strictly on context]

            ### Sources
            - [File: <file_path> (Chars: <start>-<end>)]
            """
        ).strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(
            self.model.device
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.2,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
