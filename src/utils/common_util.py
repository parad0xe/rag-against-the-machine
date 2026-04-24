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


def generate_identity(
    identity: list[str | int | bool] | None = None,
) -> str:
    if not identity:
        return ""
    return md5sum("_".join([str(v) for v in identity]))


def compute_rrf(
    ranked_lists: list[tuple[list[str], float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}

    for doc_list, weight in ranked_lists:
        for rank, doc_id in enumerate(doc_list):
            score = weight * (1.0 / (k + rank + 1))
            scores[doc_id] = scores.get(doc_id, 0.0) + score

    sorted_results = sorted(
        scores.items(), key=lambda item: item[1], reverse=True
    )

    return sorted_results


# def compute_rrf(
#    list_a: list[str],
#    list_b: list[str],
#    k: int = 60,
# ) -> list[tuple[str, float]]:
#    scores: dict[str, float] = {}
#
#    for rank, chunk_id in enumerate(list_a):
#        scores[chunk_id] = scores.get(chunk_id, 0.0) + 0.70 / (k + rank + 1)
#
#    for rank, chunk_id in enumerate(list_b):
#        scores[chunk_id] = scores.get(chunk_id, 0.0) + 0.30 / (k + rank + 1)
#
#    sorted_results = sorted(
#        scores.items(), key=lambda item: item[1], reverse=True
#    )
#
#    return sorted_results
