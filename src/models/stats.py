from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.exceptions.schema import (
    SchemaInvalidJSONFormatError,
    SchemaInvalidJSONRootError,
)


class Stats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_documents: int = 0
    num_chunks: int = 0
    num_document_by_exts: dict[str, int] = Field(default_factory=lambda: {})

    @classmethod
    def load(cls, filepath: str | Path) -> Stats:
        try:
            with open(filepath, "r") as f:
                content = json.load(f)

            if not isinstance(content, dict):
                raise SchemaInvalidJSONRootError(
                    expected=dict, context=filepath
                )
        except json.JSONDecodeError as e:
            raise SchemaInvalidJSONFormatError(
                context=filepath,
                lineno=e.lineno,
            ) from e

        return Stats.model_validate(content)

    def save(self, filepath: str | Path) -> None:
        with open(filepath, "w") as f:
            json.dump(self.model_dump(), f)
