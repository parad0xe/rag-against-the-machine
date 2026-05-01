from pathlib import Path
from typing import Literal, Protocol, TypeVar, overload

T_co = TypeVar("T_co", covariant=True)


class ReaderPort(Protocol[T_co]):
    @overload
    def read(
        self, file_path: Path, ignore_errors: Literal[False] = False
    ) -> T_co: ...

    @overload
    def read(
        self, file_path: Path, ignore_errors: Literal[True]
    ) -> T_co | None: ...

    @overload
    def read(self, file_path: Path, ignore_errors: bool) -> T_co | None: ...

    def read(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> T_co | None: ...
