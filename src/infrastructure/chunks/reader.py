import json
from pathlib import Path

from src.domain.models.base import Chunk
from src.utils.file import file_load_content


class ChunkJSONReader:
    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self._cache: dict[str, Chunk] | None = None

    def load(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not self._file_path.exists():
            return {}

        if self._cache is None:
            content = file_load_content(self._file_path, ignore_errors=True)
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
