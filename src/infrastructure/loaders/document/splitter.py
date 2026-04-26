from __future__ import annotations

from typing import Any

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.utils.file import get_extension


class LanguageTextSplitter(RecursiveCharacterTextSplitter):
    _EXTENSION_TO_LANGUAGE = {
        ".cpp": Language.CPP,
        ".hpp": Language.CPP,
        ".cc": Language.CPP,
        ".cxx": Language.CPP,
        ".go": Language.GO,
        ".java": Language.JAVA,
        ".kt": Language.KOTLIN,
        ".kts": Language.KOTLIN,
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.TS,
        ".tsx": Language.TS,
        ".php": Language.PHP,
        ".proto": Language.PROTO,
        ".py": Language.PYTHON,
        ".pyw": Language.PYTHON,
        ".r": Language.R,
        ".rst": Language.RST,
        ".rb": Language.RUBY,
        ".rs": Language.RUST,
        ".scala": Language.SCALA,
        ".sc": Language.SCALA,
        ".swift": Language.SWIFT,
        ".md": Language.MARKDOWN,
        ".markdown": Language.MARKDOWN,
        ".tex": Language.LATEX,
        ".html": Language.HTML,
        ".htm": Language.HTML,
        ".sol": Language.SOL,
        ".cs": Language.CSHARP,
        ".cob": Language.COBOL,
        ".cbl": Language.COBOL,
        ".c": Language.C,
        ".h": Language.C,
        ".lua": Language.LUA,
        ".pl": Language.PERL,
        ".pm": Language.PERL,
        ".hs": Language.HASKELL,
        ".ex": Language.ELIXIR,
        ".exs": Language.ELIXIR,
        ".ps1": Language.POWERSHELL,
        ".psm1": Language.POWERSHELL,
        ".vb": Language.VISUALBASIC6,
        ".bas": Language.VISUALBASIC6,
        ".cls": Language.VISUALBASIC6,
    }

    @classmethod
    def from_extension(
        cls, extension: str, **kwargs: Any
    ) -> LanguageTextSplitter:
        if not extension.startswith("."):
            extension = f".{extension}"
        extension = extension.lower()
        language: Language | None = cls._EXTENSION_TO_LANGUAGE.get(
            extension, None
        )

        return cls(language, **kwargs)

    @classmethod
    def from_filename(
        cls, filename: str, **kwargs: Any
    ) -> LanguageTextSplitter:
        ext = get_extension(filename)
        return cls.from_extension(ext, **kwargs)

    def __init__(self, language: Language | None, **kwargs: Any) -> None:
        try:
            separators = (
                self.get_separators_for_language(language)
                if language
                else None
            )
        except ValueError:
            separators = None
        super().__init__(separators=separators, **kwargs)
