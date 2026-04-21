import hashlib
from typing import Any, Iterable


def pluralize(word: str, count: int) -> str:
    if count <= 1:
        return word
    return word + "s"


def tolist(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, (str, bytes)):
        return [value]

    if isinstance(value, Iterable):
        return list(value)

    return [value]


def md5sum(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def file_md5sum(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
