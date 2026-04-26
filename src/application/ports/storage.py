from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.models.manifest import Manifest


class ManifestStorageInterface(ABC):
    @abstractmethod
    def save(self, file_path: Path, manifest: Manifest) -> None: ...

    @abstractmethod
    def load(
        self,
        file_path: Path,
        embedding_model_name: str,
        repositories: list[Path],
        chunk_size: int,
        fingerprint_seed: list | None = None,
    ) -> tuple[Manifest, bool]: ...
