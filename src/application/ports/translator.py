from abc import ABC, abstractmethod


class TranslatorInterface(ABC):
    @abstractmethod
    def translate_to_english(self, text: str) -> str: ...
