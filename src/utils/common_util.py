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
