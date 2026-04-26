import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.infrastructure.document.loader import load_document
from src.infrastructure.document.stores.registry import (
    IndexStoreSyncRegistry,
)
from src.infrastructure.file.loader import load_file
from src.infrastructure.manifest.manager import ManifestManager
from src.utils.file import ensure_valid_dir_path, iter_file_paths

logger = logging.getLogger(__file__)


class Indexer:
    def __init__(
        self,
        manifest_manager: ManifestManager,
        extensions: list[str],
        index_store_registry: IndexStoreSyncRegistry,
    ) -> None:
        self._manifest_manager: ManifestManager = manifest_manager
        self._index_store_registry: IndexStoreSyncRegistry = (
            index_store_registry
        )
        self._extensions: list[str] = extensions
        self._viewed_file_paths: set[Path] = set()

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
            iterator,
            desc="Indexing documents",
            bar_format=(
                "{desc} | scan {n} documents | {elapsed} | {rate_fmt}{postfix}"
            ),
            unit="files",
        ) as pbar:
            for file_path in pbar:
                pbar.set_postfix(ordered_dict=stats)

                if file_path in self._viewed_file_paths:
                    continue
                self._viewed_file_paths.add(file_path)

                file = load_file(file_path, ignore_errors=True)
                if not file:
                    continue

                cached_file = self._manifest_manager.get(file)

                document = load_document(
                    file=file,
                    chunk_size=self._manifest_manager.manifest.chunk_size,
                    cached_file=cached_file,
                )

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
                f"(add: {num_documents_to_add}) "
                "chunks"
                f"(add: {num_chunks_to_add} / delete: {num_chunks_to_delete})"
            )

    def commit(self) -> None:
        logger.info("Committing changes to stores.")

        for store in self._index_store_registry.stores:
            store.commit(self._manifest_manager.fingerprint_mismatch)

        self._manifest_manager.commit()
