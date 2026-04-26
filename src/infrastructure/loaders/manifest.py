import json
import logging
from pathlib import Path
from typing import Iterable

from src.application.ports.loader import ManifestLoaderInterface
from src.domain.exceptions.schema import (
    SchemaInvalidJSONFormatError,
    SchemaInvalidJSONRootError,
)
from src.domain.models.manifest import Manifest
from src.utils.common import compute_fingerprint
from src.utils.file import file_load_content

logger = logging.getLogger(__file__)


class ManifestJSONLoader(ManifestLoaderInterface):
    def load(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> Manifest | None:
        content = file_load_content(
            file_path,
            ignore_errors=ignore_errors,
        )
        if not content:
            return None

        try:
            manifest_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise SchemaInvalidJSONFormatError(
                context=file_path,
                lineno=e.lineno,
            ) from e

        if not isinstance(manifest_data, dict):
            if ignore_errors:
                return None
            raise SchemaInvalidJSONRootError(expected=dict, context=file_path)

        return Manifest.model_validate(manifest_data)

    def load_with_properties(
        self,
        file_path: Path,
        repositories: list[Path],
        embedding_model_name: str,
        chunk_size: int,
        fingerprint_seed: Iterable[str | int | bool] | None = None,
    ) -> tuple[Manifest, bool]:
        manifest = self.load(file_path, ignore_errors=True)
        fingerprint = compute_fingerprint(fingerprint_seed)

        if manifest:
            fingerprint_mismatch = manifest.fingerprint != fingerprint

            manifest.chunk_size = chunk_size
            manifest.repositories = repositories.copy()
            manifest.embedding_model_name = embedding_model_name
            manifest.fingerprint = fingerprint

            return manifest, fingerprint_mismatch
        else:
            new_manifest = Manifest(
                embedding_model_name=embedding_model_name,
                repositories=repositories,
                chunk_size=chunk_size,
                fingerprint=fingerprint,
            )
            return new_manifest, False
