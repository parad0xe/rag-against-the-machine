import glob
import json
import logging
from pathlib import Path
from typing import Any

from src.domain.exceptions.schema import SchemaJSONSerializationError
from src.domain.exceptions.storage import (
    StorageDirNotFoundError,
    StorageError,
    StorageFileNotFoundError,
    StorageFilePermissionError,
    StorageNotADirectoryError,
    StorageNotAFileError,
)
from src.utils.common_util import pluralize

logger = logging.getLogger(__file__)


def parse_extensions(extensions: str) -> list[str]:
    exts: list[str] = extensions.replace(" ", "").replace(".", "").split(",")
    exts = ["*" if "*" in ext else ext for ext in exts if ext]
    if "*" in exts:
        exts = ["*"]

    return exts


def get_filepaths(
    basepath: Path,
    extensions: list[str],
    recursive: bool = False,
) -> list[str]:
    logger.info(f"Loading extensions: {extensions}")

    filepaths: list[str] = []
    for ext in extensions:
        path = basepath
        if recursive:
            path = path / "**"
        path = path / f"*.{ext}"
        filepaths += glob.glob(str(path), recursive=recursive)

    logger.info(
        f"{len(filepaths)} {pluralize('file', len(filepaths))} "
        f"found for extensions {extensions}"
    )

    return filepaths


def get_extension(filepath: str) -> str:
    ext = Path(filepath).suffix
    return ext.replace(".", "")


def ensure_valide_filepath(path: Path) -> None:
    if not path.exists():
        raise StorageFileNotFoundError(path)
    if not path.is_file():
        raise StorageNotAFileError(path)


def ensure_valide_dirpath(path: Path) -> None:
    if not path.exists():
        raise StorageDirNotFoundError(path)
    if not path.is_dir():
        raise StorageNotADirectoryError(path)


def readfile(
    filepath: Path,
    ignore_unicode_error: bool = False,
) -> str | None:
    try:
        with open(filepath) as f:
            document: str = f.read()
    except FileNotFoundError as e:
        raise StorageFileNotFoundError(filepath) from e
    except PermissionError as e:
        raise StorageFilePermissionError(filepath) from e
    except UnicodeDecodeError as e:
        if ignore_unicode_error:
            return None
        raise StorageError("Invalid UTF-8 character encoding", filepath) from e
    except OSError as e:
        raise StorageError(None, filepath) from e
    return document


def file_write_json(
    filepath: Path, data: list[Any] | dict[Any, Any] | str
) -> None:
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise StorageFilePermissionError(filepath) from e
    except OSError as e:
        raise StorageError(None, filepath) from e

    try:
        str_json = (
            data if isinstance(data, str) else json.dumps(data, indent=2)
        )
    except TypeError as e:
        raise SchemaJSONSerializationError(
            reason=str(e), context=filepath
        ) from e

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(str_json)
    except FileNotFoundError as e:
        raise StorageFileNotFoundError(filepath) from e
    except PermissionError as e:
        raise StorageFilePermissionError(filepath) from e
    except OSError as e:
        raise StorageError(None, filepath) from e
