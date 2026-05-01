from pathlib import Path

from .base import RagError


class DocumentError(RagError):
    """
    Base class for all errors related to document operations.

    Attributes:
        default_message: Fallback message used when none is provided.
    """

    default_message = "A document operation failed."


class NoDocumentError(DocumentError):
    """
    Error raised when no document file is found in the system.

    Attributes:
        default_message: Fallback message used when no message is provided.
    """

    default_message = "No document file was found."

    def __init__(self, dir_path: str | Path | None = None) -> None:
        """
        Initializes the error with an optional directory path.

        Args:
            dir_path: The directory path where no document was found.
        """
        message: str | None = (
            f"No document file was found in {dir_path}"
            if dir_path is not None
            else None
        )
        super().__init__(message)
