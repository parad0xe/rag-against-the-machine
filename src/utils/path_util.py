from pathlib import Path

from src.exceptions.storage import (
    StorageDirNotFoundError,
    StorageFileNotFoundError,
    StorageNotADirectoryError,
    StorageNotAFileError,
)


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
