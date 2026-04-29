from pathlib import Path
from typing import Literal, overload

from src.domain.models.base import File
from src.utils.common import md5
from src.utils.file import file_load_content, get_extension


class LocalFileReader:
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
    ) -> File | None:
        content = file_load_content(
            file_path,
            ignore_errors=ignore_errors,
        )
        if content is None:
            return None

        str_file_path = str(file_path)

        return File(
            id=md5(str_file_path),
            path=file_path,
            ext=get_extension(str_file_path),
            hash=md5(content),
            content=content,
        )
