from pathlib import Path

from src.utils.path_util import readfile


class DocumentLoader:
    def read(self, filepath: Path) -> str | None:
        content = readfile(filepath, ignore_unicode_error=True)
        return content if content else None
