from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from src.domain.models.document import Document
from src.domain.models.file import File
from src.domain.models.manifest import Manifest, ManifestFileCache
from src.infrastructure.manifest.loader import ManifestLoader
from src.utils.file import file_write_json

logger = logging.getLogger(__file__)


class ManifestManager:
    @classmethod
    def load(
        cls,
        file_path: Path,
        repositories: list[Path],
        embedding_model_name: str,
        chunk_size: int,
        extensions: tuple[str] | list[str],
        fingerprint_seed: Iterable[str | int | bool] | None = None,
    ) -> ManifestManager:
        manifest, mismatch = ManifestLoader.load_with_properties(
            file_path=file_path,
            embedding_model_name=embedding_model_name,
            repositories=repositories,
            chunk_size=chunk_size,
            fingerprint_seed=fingerprint_seed,
        )

        return cls(file_path, manifest, mismatch, extensions)

    def __init__(
        self,
        file_path: Path,
        manifest: Manifest,
        fingerprint_mismatch: bool,
        extensions: tuple[str] | list[str],
    ) -> None:
        self._file_path: Path = file_path

        self.fingerprint_mismatch: bool = fingerprint_mismatch
        self.manifest: Manifest = manifest
        self.expired_chunk_ids: set[str] = set()

        self.__sync(extensions)

    def get(self, file: File) -> ManifestFileCache | None:
        cached_ext = self.manifest.files_by_ext.get(file.ext, {})
        return cached_ext.get(file.id, None)

    def track(self, document: Document) -> None:
        cached_file = self.get(document.file)

        if not cached_file:
            self.manifest.files_by_ext.setdefault(document.file.ext, {})

        self.manifest.files_by_ext[document.file.ext][document.file.id] = (
            ManifestFileCache(
                file_path=str(document.file.path),
                file_hash=document.file.hash,
                chunk_ids=set(document.chunk_ids),
                stores=document.stores,
            )
        )

    def commit(self) -> None:
        file_write_json(
            self._file_path, self.manifest.model_dump_json(indent=2)
        )

    def __sync(self, extensions: tuple[str] | list[str]) -> None:
        exts_to_keep, exts_to_delete = self._get_extension_sets(extensions)

        self._purge_extensions(exts_to_delete)
        self._validate_extensions(exts_to_keep)

    def _get_extension_sets(
        self, extensions: tuple[str] | list[str]
    ) -> tuple[set[str], set[str]]:
        if self.fingerprint_mismatch:
            return set(), set(self.manifest.files_by_ext.keys())

        if extensions and extensions[0] == "*":
            target_exts = set(self.manifest.files_by_ext.keys())
        else:
            target_exts = set(extensions)

        current_exts = target_exts.union(self.manifest.files_by_ext.keys())
        exts_to_keep = current_exts.intersection(target_exts)
        exts_to_delete = current_exts - target_exts

        return exts_to_keep, exts_to_delete

    def _purge_extensions(self, exts_to_delete: set[str]) -> None:
        for ext in exts_to_delete:
            if ext in self.manifest.files_by_ext:
                for cached_file in self.manifest.files_by_ext[ext].values():
                    self.expired_chunk_ids.update(cached_file.chunk_ids)
                del self.manifest.files_by_ext[ext]

    def _validate_extensions(self, exts_to_keep: set[str]) -> None:
        resolved_repos = {
            Path(repo).resolve() for repo in self.manifest.repositories
        }

        for ext in exts_to_keep:
            if ext not in self.manifest.files_by_ext:
                continue

            missing_file_ids = []
            for file_id, cached_file in self.manifest.files_by_ext[
                ext
            ].items():
                if not self._is_file_valid(cached_file, resolved_repos):
                    self.expired_chunk_ids.update(cached_file.chunk_ids)
                    missing_file_ids.append(file_id)

            for file_id in missing_file_ids:
                del self.manifest.files_by_ext[ext][file_id]

            if not self.manifest.files_by_ext[ext]:
                del self.manifest.files_by_ext[ext]

    def _is_file_valid(
        self, cached_file: ManifestFileCache, resolved_repos: set[Path]
    ) -> bool:
        cached_path = Path(cached_file.file_path)
        if not cached_path.exists():
            return False

        resolved_parents = cached_path.resolve().parents
        return any(repo in resolved_parents for repo in resolved_repos)
