import hashlib
from typing import Iterable


def md5(text: str) -> str:
    """
    Computes the MD5 hash of the given text.

    Args:
        text: The input string.

    Returns:
        The MD5 hex digest.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def compute_fingerprint(
    identity: Iterable[str | int | bool] | None = None,
) -> str:
    """
    Generates a unique fingerprint based on a sequence of values.

    Args:
        identity: An iterable of values to include in the fingerprint.

    Returns:
        The computed MD5 fingerprint.
    """
    if not identity:
        return ""

    return md5("_".join(str(v) for v in identity))


def parse_extensions(extensions: str) -> list[str]:
    """
    Parses a comma-separated string of file extensions.

    Args:
        extensions: Raw extension string (e.g., '.py, .js').

    Returns:
        A list of cleaned extension strings.
    """
    exts: list[str] = extensions.replace(" ", "").replace(".", "").split(",")
    exts = ["*" if "*" in ext else ext for ext in exts if ext]
    if "*" in exts:
        exts = ["*"]

    return exts
