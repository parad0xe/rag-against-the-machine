import json
from pathlib import Path

from src.domain.models.chunk import Chunk
from src.utils.path_util import readfile


class ChunksRepository:
    def __init__(self, filepath: Path) -> None:
        self._filepath = filepath
        self._cache: dict[str, Chunk] | None = None

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not self._filepath.exists():
            return {}

        if self._cache is None:
            content = readfile(self._filepath, ignore_unicode_error=True)
            if not content:
                return {}
            try:
                self._cache = json.loads(content)
            except json.JSONDecodeError:
                return {}

        if self._cache is None:
            return {}

        return {
            chunk_id: self._cache[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self._cache
        }
