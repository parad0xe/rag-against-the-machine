from pydantic import BaseModel, ConfigDict, Field

from src.domain.models.chunk import Chunk
from src.domain.models.file import File


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: File

    chunks: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_metadatas: dict[str, Chunk] = Field(default_factory=dict)

    stores: set[str] = Field(default_factory=set)
