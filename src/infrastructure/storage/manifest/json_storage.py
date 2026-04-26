from pathlib import Path

from src.application.ports.storage import ManifestStorageInterface
from src.domain.models.manifest import Manifest
from src.infrastructure.loaders.manifest import ManifestJSONLoader
from src.utils.file import file_write_json


class ManifestJSONStorage(ManifestStorageInterface):
    def save(self, file_path: Path, manifest: Manifest) -> None:
        file_write_json(file_path, manifest.model_dump_json(indent=2))

    def load(
        self,
        file_path: Path,
        embedding_model_name: str,
        repositories: list[Path],
        chunk_size: int,
        fingerprint_seed: list | None = None,
    ) -> tuple[Manifest, bool]:
        return ManifestJSONLoader().load_with_properties(
            file_path=file_path,
            embedding_model_name=embedding_model_name,
            repositories=repositories,
            chunk_size=chunk_size,
            fingerprint_seed=fingerprint_seed,
        )
