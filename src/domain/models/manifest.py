from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class ManifestFileCache(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    file_hash: str
    chunk_ids: set[str]
    stores: set[str]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model_name: str
    with_semantic: bool
    repositories: list[Path]
    chunk_size: PositiveInt
    files_by_ext: dict[str, dict[str, ManifestFileCache]] = Field(
        default_factory=dict
    )
    fingerprint: str
