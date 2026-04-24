from enum import Enum, auto

from pydantic import BaseModel, ConfigDict, Field

from src.domain.models.chunk import Chunk


class DocumentStatus(str, Enum):
    NOTHING_TO_DO = auto()
    NEW = auto()
    UPDATE = auto()


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    hash: str
    ext: str

    filepath: str
    content: str

    chunks: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_metadatas: dict[str, Chunk] = Field(default_factory=dict)
