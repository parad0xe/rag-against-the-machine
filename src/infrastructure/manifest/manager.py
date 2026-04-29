from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from src.application.ports.manifest import ManifestRepositoryPort
from src.domain.models.base import Document, File, Manifest, ManifestFileCache
from src.utils.common import compute_fingerprint

logger = logging.getLogger(__file__)


class ManifestManager:
    @property
    def manifest(self) -> Manifest:
        return self._manifest

    @property
    def expired_chunk_ids(self) -> set[str]:
        return self._expired_chunk_ids

    @property
    def fingerprint_mismatch(self) -> bool:
        return self._fingerprint_mismatch

    def __init__(
        self,
        file_path: Path,
        manifest_repository: ManifestRepositoryPort,
        extensions: tuple[str] | list[str],
        embedding_model_name: str,
        repositories: list[Path],
        chunk_size: int,
        with_semantic: bool,
        fingerprint_seed: Iterable[str | int | bool] | None = None,
    ) -> None:
        self._file_path = file_path
        self._manifest_repository = manifest_repository
        self._expired_chunk_ids: set[str] = set()

        loaded_manifest = manifest_repository.load(
            file_path, ignore_errors=True
        )

        current_fingerprint = compute_fingerprint(fingerprint_seed)

        if loaded_manifest:
            self._fingerprint_mismatch = (
                loaded_manifest.fingerprint != current_fingerprint
            )
            self._manifest = loaded_manifest
            self._manifest.embedding_model_name = embedding_model_name
            self._manifest.with_semantic = with_semantic
            self._manifest.repositories = repositories.copy()
            self._manifest.chunk_size = chunk_size
            self._manifest.fingerprint = current_fingerprint
        else:
            self._fingerprint_mismatch = False
            self._manifest = Manifest(
                embedding_model_name=embedding_model_name,
                with_semantic=with_semantic,
                repositories=repositories.copy(),
                chunk_size=chunk_size,
                fingerprint=current_fingerprint,
            )

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
        self._manifest_repository.save(self._file_path, self.manifest)

    def __sync(self, extensions: tuple[str] | list[str]) -> None:
        if self._fingerprint_mismatch:
            self._purge_extensions(set(self.manifest.files_by_ext.keys()))
            return

        target_exts = (
            set(self.manifest.files_by_ext.keys())
            if extensions and extensions[0] == "*"
            else set(extensions)
        )

        current_exts = set(self.manifest.files_by_ext.keys())
        exts_to_keep = current_exts.intersection(target_exts)
        exts_to_delete = current_exts - target_exts

        self._purge_extensions(exts_to_delete)
        self._validate_extensions(exts_to_keep)

    def _purge_extensions(self, exts_to_delete: set[str]) -> None:
        for ext in exts_to_delete:
            if ext in self.manifest.files_by_ext:
                for cached_file in self.manifest.files_by_ext[ext].values():
                    self._expired_chunk_ids.update(cached_file.chunk_ids)
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
                    self._expired_chunk_ids.update(cached_file.chunk_ids)
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
