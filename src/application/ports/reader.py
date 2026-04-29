from pathlib import Path
from typing import Literal, Protocol, overload

from src.domain.models.base import (
    Chunk,
    File,
)
from src.domain.models.dataset import RagDataset


class FileReaderPort(Protocol):
    @overload
    def read(
        self, file_path: Path, ignore_errors: Literal[False] = False
    ) -> File: ...

    @overload
    def read(
        self, file_path: Path, ignore_errors: Literal[True]
    ) -> File | None: ...

    @overload
    def read(self, file_path: Path, ignore_errors: bool) -> File | None: ...

    def read(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> File | None: ...


class RagDatasetReaderPort(Protocol):
    @overload
    def read(
        self, file_path: Path, ignore_errors: Literal[False] = False
    ) -> RagDataset: ...

    @overload
    def read(
        self, file_path: Path, ignore_errors: Literal[True]
    ) -> RagDataset | None: ...

    @overload
    def read(
        self, file_path: Path, ignore_errors: bool
    ) -> RagDataset | None: ...

    def read(
        self, file_path: Path, ignore_errors: bool = False
    ) -> RagDataset | None: ...


class ChunkReaderPort(Protocol):
    def load(self, chunk_ids: list[str]) -> dict[str, Chunk]: ...
