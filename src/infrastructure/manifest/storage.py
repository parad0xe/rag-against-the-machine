import json
import logging
from pathlib import Path
from typing import Literal, overload

from src.application.ports.manifest import ManifestStoragePort
from src.domain.exceptions.schema import (
    SchemaInvalidJSONFormatError,
)
from src.domain.models.base import Manifest
from src.utils.file import file_load_content, file_write_json

logger = logging.getLogger(__file__)


class ManifestJSONStorage(ManifestStoragePort):
    """
    Storage implementation that reads and writes the manifest as JSON.
    """

    def save(self, file_path: Path, manifest: Manifest) -> None:
        """
        Saves the manifest model to a JSON file.

        Args:
            file_path: Destination path.
            manifest: The manifest model to save.
        """
        file_write_json(file_path, manifest.model_dump_json(indent=2))

    @overload
    def read(
        self, file_path: Path, ignore_errors: Literal[False] = False
    ) -> Manifest: ...

    @overload
    def read(
        self, file_path: Path, ignore_errors: Literal[True]
    ) -> Manifest | None: ...

    @overload
    def read(
        self, file_path: Path, ignore_errors: bool
    ) -> Manifest | None: ...

    def read(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> Manifest | None:
        """
        Reads and parses the manifest from a JSON file.

        Args:
            file_path: Path to the manifest file.
            ignore_errors: Whether to suppress errors on failure.

        Returns:
            The Manifest model or None.

        Raises:
            SchemaInvalidJSONFormatError: If the manifest JSON is malformed.
        """
        content = file_load_content(
            file_path,
            ignore_errors=ignore_errors,
        )
        if not content:
            return None

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise SchemaInvalidJSONFormatError(
                context=file_path,
                lineno=e.lineno,
            ) from e

        if not isinstance(data, dict):
            return None

        return Manifest.model_validate(data)
