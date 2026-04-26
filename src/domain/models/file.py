from pathlib import Path

from pydantic import BaseModel, ConfigDict


class File(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    path: Path
    ext: str
    hash: str
    content: str
