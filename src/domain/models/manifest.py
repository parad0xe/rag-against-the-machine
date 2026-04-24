from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class RawManifestFileCacheDict(TypedDict):
    file_path: str
    file_hash: str
    chunk_ids: set[str]


class RawManifestDict(TypedDict, total=False):
    embedding_model_name: str
    repositories: list[str]
    chunk_size: int
    files_by_ext: dict[str, dict[str, RawManifestFileCacheDict]]
    identity: str


class ManifestFileCache(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    file_hash: str
    chunk_ids: set[str]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model_name: str
    repositories: list[Path]
    chunk_size: PositiveInt
    files_by_ext: dict[str, dict[str, ManifestFileCache]] = Field(
        default_factory=dict
    )
    identity: str
