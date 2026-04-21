from pathlib import Path

from src.exceptions.base import RagError


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

    def __init__(self, dirpath: str | Path | None = None) -> None:
        """
        Initializes the error with an optional directory path.

        Args:
            dirpath: The directory path where no document was found.
        """
        message: str | None = (
            f"No document file was found in {dirpath}"
            if dirpath is not None
            else None
        )
        super().__init__(message)
