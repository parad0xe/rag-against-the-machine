from pathlib import Path
from typing import Literal, Protocol, TypeVar, overload

T_co = TypeVar("T_co", covariant=True)


class ReaderPort(Protocol[T_co]):
    """
    Generic port for reading data into a specific model.
    """

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
    ) -> T_co | None:
        """
        Read and parse data from the specified file.

        Args:
            file_path: The path to the file to read.
            ignore_errors: Whether to suppress errors and return None.

        Returns:
            The parsed model or None on failure.
        """
        ...
