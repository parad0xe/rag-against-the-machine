import json
from pathlib import Path
from typing import Any

from src.domain.exceptions.schema import (
    SchemaInvalidJSONFormatError,
    SchemaInvalidJSONRootError,
)
from src.domain.models.manifest import Manifest


class ManifestStorage:
    """
    Handles the persistence of the Manifest configuration.
    """

    def __init__(self, filepath: Path) -> None:
        """
        Initializes the storage.

        Args:
            filepath: Location of the manifest file.
        """
        self._filepath = filepath

    def load(self) -> dict[str, Any] | None:
        """
        Loads the manifest data from a JSON file.

        Returns:
            The parsed JSON data as a dictionary, or None if missing.
        """
        if not self._filepath.exists():
            return None

        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if not content:
                return None

            manifest_data = json.loads(content)

            if not isinstance(manifest_data, dict):
                raise SchemaInvalidJSONRootError(
                    expected=dict, context=self._filepath
                )

            return manifest_data
        except json.JSONDecodeError as e:
            raise SchemaInvalidJSONFormatError(
                context=self._filepath,
                lineno=e.lineno,
            ) from e

    def save(self, manifest: Manifest) -> None:
        """
        Saves the manifest state to the filesystem.

        Args:
            manifest: Instance of the manifest to serialize.
        """
        self._filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(self._filepath, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
