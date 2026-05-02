from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class Chunk(TypedDict):
    """
    Metadata and content for a specific text chunk.

    Attributes:
        text: The raw text content of the chunk.
        hash: MD5 hash of the chunk text.
        file_path: Relative path to the source file.
        first_character_index: Start position in the source file.
        last_character_index: End position in the source file.
    """

    text: str
    hash: str
    file_path: str
    first_character_index: int
    last_character_index: int


class File(BaseModel):
    """
    Representation of a source file in the repository.

    Attributes:
        id: Unique identifier (MD5 hash of the file path).
        path: Absolute path to the file on disk.
        ext: File extension without the leading dot.
        hash: MD5 hash of the file content.
        content: Raw text content of the file.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    path: Path
    ext: str
    hash: str
    content: str


class Document(BaseModel):
    """
    A file processed into chunks for indexing.

    Attributes:
        file: The source file model.
        chunks: List of raw text chunks.
        chunk_ids: List of unique identifiers for each chunk.
        chunk_metadatas: Mapping of chunk IDs to their metadata.
        stores: Set of store names where this document is indexed.
    """

    model_config = ConfigDict(extra="forbid")

    file: File
    chunks: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_metadatas: dict[str, Chunk] = Field(default_factory=dict)
    stores: set[str] = Field(default_factory=set)


class ManifestFileCache(BaseModel):
    """
    Cached file information stored in the manifest.

    Attributes:
        file_path: Relative path to the file.
        file_hash: MD5 hash of the file content.
        chunk_ids: Set of chunk IDs associated with this file.
        stores: Set of store names where this file is currently indexed.
    """

    model_config = ConfigDict(extra="forbid")

    file_path: str
    file_hash: str
    chunk_ids: set[str]
    stores: set[str]


class Manifest(BaseModel):
    """
    Global system state and configuration for the index.

    Attributes:
        embedding_model_name: Name of the model used for embeddings.
        with_semantic: Whether semantic search is enabled.
        repositories: List of repository paths being indexed.
        chunk_size: Maximum size of each text chunk.
        files_by_ext: Mapping of extensions to file caches.
        fingerprint: Unique identifier for the current configuration.
    """

    model_config = ConfigDict(extra="forbid")

    embedding_model_name: str
    with_semantic: bool
    repositories: list[Path]
    chunk_size: PositiveInt
    files_by_ext: dict[str, dict[str, ManifestFileCache]] = Field(
        default_factory=dict
    )
    fingerprint: str
