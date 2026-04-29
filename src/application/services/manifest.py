from pathlib import Path
from typing import TypedDict

from src.application.ports.manifest import ManifestStoragePort


class ExtensionStat(TypedDict):
    file_count: int
    chunk_count: int


class ManifestStats(TypedDict):
    embedding_model: str
    with_semantic: bool
    chunk_size: int
    total_files: int
    total_chunks: int
    repositories: list[str]
    extensions: dict[str, ExtensionStat]
    fingerprint: str


class ManifestService:
    def __init__(self, storage: ManifestStoragePort) -> None:
        self._storage = storage

    def get_stats(self, file_path: Path) -> ManifestStats:
        manifest = self._storage.read(file_path, ignore_errors=False)

        total_files = 0
        total_chunks = 0
        ext_stats: dict[str, ExtensionStat] = {}

        for ext, files in manifest.files_by_ext.items():
            files_count = len(files)
            chunks_count = sum(len(f.chunk_ids) for f in files.values())

            total_files += files_count
            total_chunks += chunks_count

            ext_stats[ext] = ExtensionStat(
                file_count=files_count, chunk_count=chunks_count
            )

        return ManifestStats(
            embedding_model=manifest.embedding_model_name,
            with_semantic=manifest.with_semantic,
            chunk_size=manifest.chunk_size,
            total_files=total_files,
            total_chunks=total_chunks,
            repositories=[str(p) for p in manifest.repositories],
            extensions=ext_stats,
            fingerprint=manifest.fingerprint,
        )
