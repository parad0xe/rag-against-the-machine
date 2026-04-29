import logging
from pathlib import Path

from tqdm.rich import tqdm

from src.application.ports.index_store import (
    IndexStoreRegistryPort,
    IndexStoreSyncPort,
)
from src.application.ports.loader import DocumentLoaderPort
from src.application.ports.manifest import ManifestManagerPort
from src.application.ports.reader import (
    FileReaderPort,
)
from src.utils.file import ensure_valid_dir_path, iter_file_paths

logger = logging.getLogger(__file__)


class Indexer:
    def __init__(
        self,
        manifest_manager: ManifestManagerPort,
        extensions: list[str],
        index_store_registry: IndexStoreRegistryPort[IndexStoreSyncPort],
        file_loader: FileReaderPort,
        document_loader: DocumentLoaderPort,
    ) -> None:
        self._manifest_manager = manifest_manager
        self._index_store_registry = index_store_registry
        self._extensions: list[str] = extensions
        self._file_loader: FileReaderPort = file_loader
        self._document_loader: DocumentLoaderPort = document_loader
        self._viewed_file_paths: set[Path] = set()
        self.founded_documents: int = 0

        for store in index_store_registry.stores:
            store.delete(manifest_manager.expired_chunk_ids)

    def index(self, repository: Path) -> None:
        logger.info(f"Starting indexing for repository: {repository}")
        ensure_valid_dir_path(repository)

        iterator = iter_file_paths(
            repository,
            self._extensions,
            recursive=True,
        )

        if iterator is None:
            logger.warning(f"Invalid repository path: {repository}. Skipped")
            return

        for file_path in tqdm(
            list(iterator), desc="Indexing documents", unit="files"
        ):
            if file_path in self._viewed_file_paths:
                continue
            self._viewed_file_paths.add(file_path)

            file = self._file_loader.read(file_path, ignore_errors=True)
            if not file or not file.content:
                continue

            cached_file = self._manifest_manager.get(file)
            document = self._document_loader.load(
                file=file,
                chunk_size=self._manifest_manager.manifest.chunk_size,
                cached_file=cached_file,
            )

            self.founded_documents += 1

            for store in self._index_store_registry.stores:
                store.track(document, cached_file=cached_file)

            self._manifest_manager.track(document)

    def commit(self) -> None:
        logger.info("Committing changes to stores.")

        for store in self._index_store_registry.stores:
            store.commit(
                require_reset=self._manifest_manager.fingerprint_mismatch
            )

        self._manifest_manager.commit()
