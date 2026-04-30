import logging
from threading import Thread
from typing import Any, Generator, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from transformers.utils import logging as transformers_logging

from src.application.ports.llm.engine import (
    ChatMessage,
    TextGenerationEnginePort,
)
from src.config import settings
from src.domain.exceptions.base import RagError

cast(Any, transformers_logging).disable_progress_bar()

logger = logging.getLogger(__file__)


class HuggingFaceCausalEngine(TextGenerationEnginePort):
    def __init__(self, model_name: str = settings.llm_model) -> None:
        logger.info(f"Loading LLM {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="sdpa",
            )
            cast(Any, self.model).eval()
        except MemoryError as e:
            raise RagError("Failed to load LLM into memory") from e

    def generate(
        self,
        messages: list[ChatMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Generator[str, None, None]:
        text = self.tokenizer.apply_chat_template(
            cast(Any, messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=kwargs.pop("enable_thinking", False),
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=8192,
            truncation=True,
        ).to(self.model.device)

        if stream:
            return self._generate_stream(inputs, **kwargs)
        return self._generate_sync(inputs, **kwargs)

    def _generate_stream(
        self,
        inputs: Any,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
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
            yield str(new_text)

    def _generate_sync(self, inputs: Any, **kwargs: Any) -> str:
        generation_kwargs = dict(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        try:
            with torch.inference_mode():
                outputs = cast(Any, self.model).generate(**generation_kwargs)

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]

            result = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            return str(result).strip()

        except Exception as e:
            logger.error(
                f"LLM generation failed ({e.__class__.__name__}): {e}"
            )
            return ""
