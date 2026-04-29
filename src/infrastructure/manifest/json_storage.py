from pathlib import Path

from src.domain.models.base import Manifest
from src.infrastructure.manifest.json_loader import ManifestJSONLoader
from src.utils.file import file_write_json


class ManifestJSONStorage:
    def save(self, file_path: Path, manifest: Manifest) -> None:
        file_write_json(file_path, manifest.model_dump_json(indent=2))

    def load(
        self,
        file_path: Path,
        embedding_model_name: str,
        repositories: list[Path],
        chunk_size: int,
        with_semantic: bool,
        fingerprint_seed: list[str | int | bool] | None = None,
    ) -> tuple[Manifest, bool]:
        return ManifestJSONLoader().load_with_properties(
            file_path=file_path,
            embedding_model_name=embedding_model_name,
            repositories=repositories,
            chunk_size=chunk_size,
            with_semantic=with_semantic,
            fingerprint_seed=fingerprint_seed,
        )
