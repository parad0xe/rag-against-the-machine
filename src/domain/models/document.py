from enum import Enum, auto
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(str, Enum):
    NOTHING_TO_DO = auto()
    NEW = auto()
    UPDATE = auto()


class ChunkMetadata(TypedDict):
    text: str
    hash: str
    file_path: str
    first_character_index: int
    last_character_index: int


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    hash: str
    ext: str

    filepath: str
    content: str

    chunks: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_metadatas: dict[str, ChunkMetadata] = Field(default_factory=dict)
