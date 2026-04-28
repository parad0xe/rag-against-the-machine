from abc import ABC, abstractmethod
from typing import Generator


class LlmInterface(ABC):
    @abstractmethod
    def generate_answer(
        self, query: str, context: str
    ) -> Generator[str, None, None]: ...
