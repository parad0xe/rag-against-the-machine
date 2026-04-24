from abc import ABC, abstractmethod

from src.domain.models.manifest import Manifest


class BaseManifestStorage(ABC):
    @abstractmethod
    def load(self) -> tuple[Manifest, bool]: ...

    @abstractmethod
    def save(self, manifest: Manifest) -> None: ...
