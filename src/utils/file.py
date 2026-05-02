import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Generator, Iterable

from src.domain.exceptions.schema import SchemaJSONSerializationError
from src.domain.exceptions.storage import (
    StorageDirNotFoundError,
    StorageError,
    StorageFileNotFoundError,
    StorageFilePermissionError,
    StorageNotADirectoryError,
    StorageNotAFileError,
)

logger = logging.getLogger(__file__)


def iter_file_paths(
    basepath: Path,
    extensions: list[str],
    recursive: bool = False,
) -> Generator[Path, None, None] | None:
    """
    Iterates through files in a directory matching specific extensions.

    Args:
        basepath: Root directory to scan.
        extensions: List of extensions to include.
        recursive: Whether to scan subdirectories.

    Yields:
        A Path object for each matching file.
    """
    if not basepath.exists():
        return None

    for ext in extensions:
        pattern = f"*.{ext}"

        path_generator = (
            basepath.rglob(pattern) if recursive else basepath.glob(pattern)
        )

        for path in path_generator:
            if path.is_file() and path.suffix.strip():
                yield Path(path)

    return None


def ensure_valid_file_path(path: Path | str) -> None:
    """
    Validates that a path exists and is a file.

    Args:
        path: The path to check.

    Raises:
        StorageFileNotFoundError: If the path does not exist.
        StorageNotAFileError: If the path is not a file.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise StorageFileNotFoundError(path)
    if not path.is_file():
        raise StorageNotAFileError(path)


def ensure_valid_dir_path(
    path: Path | str,
    modes: Iterable[int] = (os.R_OK,),
) -> None:
    """
    Validates that a path exists, is a directory, and has correct permissions.

    Args:
        path: The path to check.
        modes: Sequence of access modes to verify (e.g., os.R_OK).

    Raises:
        StorageDirNotFoundError: If the path does not exist.
        StorageNotADirectoryError: If the path is not a directory.
        StorageFilePermissionError: If access modes are not met.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise StorageDirNotFoundError(path)

    if not path.is_dir():
        raise StorageNotADirectoryError(path)

    for mode in modes:
        if not os.access(path, mode):
            raise StorageFilePermissionError(path)


def get_extension(file_path: str) -> str:
    """
    Extracts the file extension without the leading dot.

    Args:
        file_path: Path string.

    Returns:
        The extension string.
    """
    ext = Path(file_path).suffix
    return ext.replace(".", "")


def file_load_content(
    file_path: Path,
    ignore_errors: bool = False,
) -> str | None:
    """
    Reads the content of a text file using UTF-8 encoding.

    Args:
        file_path: Path to the file.
        ignore_errors: Whether to suppress IO and encoding errors.

    Returns:
        The raw text content or None.

    Raises:
        StorageFileNotFoundError: If file not found and ignore_errors is False.
        StorageFilePermissionError: If permission denied.
        StorageError: For invalid encoding or other OS errors.
    """
    try:
        with open(file_path, encoding="utf-8", errors="strict") as f:
            return f.read()
    except (OSError, UnicodeDecodeError) as e:
        if ignore_errors:
            logger.warning(
                f"failed to load file {'/'.join(file_path.parts[-2:])}: {e}"
            )
            return None

        if isinstance(e, FileNotFoundError):
            raise StorageFileNotFoundError(file_path) from e

        if isinstance(e, PermissionError):
            raise StorageFilePermissionError(file_path) from e

        if isinstance(e, UnicodeDecodeError):
            raise StorageError(
                "Invalid UTF-8 character encoding", file_path
            ) from e

        raise StorageError(None, file_path) from e


def file_write_json(
    file_path: Path, data: list[Any] | dict[Any, Any] | str
) -> None:
    """
    Writes data to a JSON file.

    Args:
        file_path: Destination path.
        data: Data structure or raw JSON string to write.

    Raises:
        StorageFilePermissionError: If parent dir cannot be created or file
            cannot be written.
        SchemaJSONSerializationError: If data cannot be serialized.
        StorageFileNotFoundError: If the file path is invalid.
        StorageError: For other OS-level errors.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise StorageFilePermissionError(file_path) from e
    except OSError as e:
        raise StorageError(None, file_path) from e

    try:
        str_json = (
            data if isinstance(data, str) else json.dumps(data, indent=2)
        )
    except TypeError as e:
        raise SchemaJSONSerializationError(
            reason=str(e), context=file_path
        ) from e

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str_json)
    except FileNotFoundError as e:
        raise StorageFileNotFoundError(file_path) from e
    except PermissionError as e:
        raise StorageFilePermissionError(file_path) from e
    except OSError as e:
        raise StorageError(None, file_path) from e


def safe_rmtree(dir_path: Path) -> None:
    """
    Safely deletes a directory tree, restricted to the data directory.

    Args:
        dir_path: The directory to remove.

    Raises:
        StorageError: If the path is outside of the configured data directory.
    """
    from src.config import settings

    resolved_path = dir_path.resolve()
    resolved_base = settings.data_dir.resolve()

    if not resolved_path.is_relative_to(resolved_base):
        raise StorageError(
            f"Security violation: deletion forbidden outside of "
            f"'{resolved_base}'",
            dir_path,
        )

    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
