import hashlib
from typing import Iterable


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def compute_fingerprint(
    identity: Iterable[str | int | bool] | None = None,
) -> str:
    if not identity:
        return ""

    return md5("_".join(str(v) for v in identity))


def parse_extensions(extensions: str) -> list[str]:
    exts: list[str] = extensions.replace(" ", "").replace(".", "").split(",")
    exts = ["*" if "*" in ext else ext for ext in exts if ext]
    if "*" in exts:
        exts = ["*"]

    return exts
