import json
import logging
import os
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
    if not basepath.exists():
        return None

    for ext in extensions:
        pattern = f"*.{ext}"

        path_generator = (
            basepath.rglob(pattern) if recursive else basepath.glob(pattern)
        )

        for path in path_generator:
            if path.is_file():
                yield Path(path)

    return None


def ensure_valid_file_path(path: Path) -> None:
    if not path.exists():
        raise StorageFileNotFoundError(path)
    if not path.is_file():
        raise StorageNotAFileError(path)


def ensure_valid_dir_path(
    path: Path, modes: Iterable[int] = (os.R_OK,)
) -> None:
    if not path.exists():
        raise StorageDirNotFoundError(path)

    if not path.is_dir():
        raise StorageNotADirectoryError(path)

    for mode in modes:
        if not os.access(path, mode):
            raise StorageFilePermissionError(path)


def get_extension(file_path: str) -> str:
    ext = Path(file_path).suffix
    return ext.replace(".", "")


def file_load_content(
    file_path: Path,
    ignore_errors: bool = False,
) -> str | None:
    try:
        with open(file_path, encoding="utf-8", errors="strict") as f:
            return f.read()
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(f"failed to load file {file_path}: {e}")
        if ignore_errors:
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
