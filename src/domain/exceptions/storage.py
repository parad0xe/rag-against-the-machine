from pathlib import Path
from typing import Optional, Union

from .base import RagError


class StorageError(RagError):
    """
    Base class for all errors related to file operations (read/write).

    Attributes:
        default_message: Fallback message used when no message is provided.
        file_path: The path to the file that caused the error, if applicable.
    """

    default_message = "A file operation failed."

    def __init__(
        self,
        message: Optional[str] = None,
        file_path: Union[str | Path, None] = None,
    ) -> None:
        """
        Initializes the error with an optional message and file_path.

        Args:
            message: Custom error message.
            file_path: Path to the file.
        """
        self.file_path = file_path

        if message is None:
            message = (
                f"Failed to access or process file '{file_path}'."
                if file_path
                else self.default_message
            )
        elif self.file_path:
            message = f"({self.file_path}) {message}"

        super().__init__(message)


class StorageDirNotFoundError(StorageError):
    """
    Error raised when the target path does not exist on the filesystem.
    """

    def __init__(self, dir_path: str | Path) -> None:
        super().__init__(
            f"The specified directory '{dir_path}' was not found.",
            file_path=dir_path,
        )


class StorageFileNotFoundError(StorageError):
    """
    Error raised when the target file does not exist on the filesystem.
    """

    def __init__(self, file_path: str | Path) -> None:
        super().__init__(
            f"The specified file '{file_path}' was not found.",
            file_path=file_path,
        )


class StorageFilePermissionError(StorageError):
    """Error raised when lacking permissions to read or write the file."""

    def __init__(self, file_path: str | Path) -> None:
        super().__init__(
            f"Permission denied for the specified file '{file_path}'.",
            file_path=file_path,
        )


class StorageEmptyFileError(StorageError):
    """
    Error raised when the provided file contains no data.
    """

    def __init__(self, file_path: str | Path) -> None:
        super().__init__(
            f"The specified file '{file_path}' is empty.",
            file_path=file_path,
        )


class StorageNotAFileError(StorageError):
    """
    Error raised when the specified path exists but is not a file.
    """

    def __init__(self, file_path: str | Path) -> None:
        """
        Initializes the error with the problematic path.

        Args:
            file_path: Path that is expected to be a file.
        """
        super().__init__(
            f"The specified path '{file_path}' is not a file.",
            file_path=file_path,
        )


class StorageNotADirectoryError(StorageError):
    """
    Error raised when the specified path exists but is not a directory.
    """

    def __init__(self, dir_path: str | Path) -> None:
        """
        Initializes the error with the problematic path.

        Args:
            dir_path: Path that is expected to be a directory.
        """
        super().__init__(
            f"The specified path '{dir_path}' is not a directory.",
            file_path=dir_path,
        )
