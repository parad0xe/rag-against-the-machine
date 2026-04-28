from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class Chunk(TypedDict):
    text: str
    hash: str
    file_path: str
    first_character_index: int
    last_character_index: int


class File(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    path: Path
    ext: str
    hash: str
    content: str


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: File
    chunks: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_metadatas: dict[str, Chunk] = Field(default_factory=dict)
    stores: set[str] = Field(default_factory=set)


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
