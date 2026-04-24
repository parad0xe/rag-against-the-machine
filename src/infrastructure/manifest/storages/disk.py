import json
from pathlib import Path

from src.domain.exceptions.schema import (
    SchemaInvalidJSONFormatError,
    SchemaInvalidJSONRootError,
)
from src.domain.models.manifest import Manifest
from src.infrastructure.manifest.storages.base import BaseManifestStorage
from src.utils.common_util import generate_identity
from src.utils.path_util import file_write_json, readfile


class ManifestDiskStorage(BaseManifestStorage):
    def __init__(
        self,
        filepath: Path,
        repositories: list[Path],
        embedding_model_name: str,
        chunk_size: int,
        identity: list[str | int | bool] | None = None,
    ) -> None:
        super().__init__()
        self._filepath: Path = filepath
        self._repositories = repositories
        self._embedding_model_name = embedding_model_name
        self._chunk_size = chunk_size
        self._identity = generate_identity(identity)

    def load(self) -> tuple[Manifest, bool]:
        default_data = {
            "repositories": [str(r) for r in self._repositories],
            "chunk_size": self._chunk_size,
            "embedding_model_name": self._embedding_model_name,
            "identity": self._identity,
        }

        if not self._filepath.exists():
            new_manifest = Manifest.model_validate(default_data)
            return new_manifest, False

        content = readfile(self._filepath)

        if not content:
            new_manifest = Manifest.model_validate(default_data)
            return new_manifest, False

        try:
            manifest_data = json.loads(content)

            if not isinstance(manifest_data, dict):
                raise SchemaInvalidJSONRootError(
                    expected=dict, context=self._filepath
                )
        except json.JSONDecodeError as e:
            raise SchemaInvalidJSONFormatError(
                context=self._filepath,
                lineno=e.lineno,
            ) from e

        manifest = Manifest.model_validate(manifest_data)
        identity_mismatch = manifest.identity != self._identity

        merged_data = {**manifest_data, **default_data}
        merged_manifest = Manifest.model_validate(merged_data)

        return merged_manifest, identity_mismatch

    def save(self, manifest: Manifest) -> None:
        file_write_json(self._filepath, manifest.model_dump_json(indent=2))
