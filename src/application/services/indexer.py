import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from tqdm.rich import tqdm

from src.application.ports.index_store.registry import (
    IndexStoreRegistryInterface,
)
from src.application.ports.index_store.store import IndexStoreSyncInterface
from src.application.ports.loader import (
    DocumentLoaderInterface,
    FileLoaderInterface,
)
from src.infrastructure.storage.manifest.manager import ManifestManager
from src.utils.file import ensure_valid_dir_path, iter_file_paths

logger = logging.getLogger(__file__)


class Indexer:
    def __init__(
        self,
        manifest_manager: ManifestManager,
        extensions: list[str],
        index_store_registry: IndexStoreRegistryInterface[
            IndexStoreSyncInterface
        ],
        file_loader: FileLoaderInterface,
        document_loader: DocumentLoaderInterface,
    ) -> None:
        self._manifest_manager: ManifestManager = manifest_manager
        self._index_store_registry = index_store_registry
        self._extensions: list[str] = extensions
        self._file_loader: FileLoaderInterface = file_loader
        self._document_loader: DocumentLoaderInterface = document_loader
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

        stats: OrderedDict[str, Any] = OrderedDict(
            documents=0,
            chunks=0,
        )

        with tqdm(
            list(iterator),
            desc="Indexing documents",
            unit="files",
        ) as pbar:
            for file_path in pbar:
                pbar.set_postfix(ordered_dict=stats)

                if file_path in self._viewed_file_paths:
                    continue
                self._viewed_file_paths.add(file_path)

                file = self._file_loader.load(file_path, ignore_errors=True)
                if not file:
                    continue

                cached_file = self._manifest_manager.get(file)
                document = self._document_loader.load(
                    file=file,
                    chunk_size=self._manifest_manager.manifest.chunk_size,
                    cached_file=cached_file,
                )

                self.founded_documents += 1
                stats["documents"] += 1
                stats["chunks"] += len(document.chunk_ids)

                for store in self._index_store_registry.stores:
                    store.track(document, cached_file=cached_file)

                self._manifest_manager.track(document)

                pbar.set_postfix(ordered_dict=stats)

        for store in self._index_store_registry.stores:
            num_chunks_to_delete = len(store._delete_chunk_ids)
            num_chunks_to_add = sum(
                [len(x.chunk_ids) for x in store._add_documents]
            )
            num_documents_to_add = len(store._add_documents)
            logger.debug(
                f"[{store.__class__.__name__}] "
                "documents"
                f"(to add: {num_documents_to_add}) "
                "chunks"
                f"(to add: {num_chunks_to_add} "
                f"/ to delete: {num_chunks_to_delete})"
            )

    def commit(self) -> None:
        logger.info("Committing changes to stores.")

        for store in self._index_store_registry.stores:
            store.commit(
                require_reset=self._manifest_manager.fingerprint_mismatch
            )

        self._manifest_manager.commit()
