from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from src.exceptions.schema import (
    SchemaInvalidJSONFormatError,
    SchemaInvalidJSONRootError,
)


class CachedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filepath: str
    filehash: str
    chunk_ids: set[str]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    chunk_size: PositiveInt = 0
    extensions: set[str] = Field(default_factory=lambda: set())
    already_exists: bool = False
    files_by_ext: dict[str, dict[str, CachedFile]] = Field(
        default_factory=lambda: {}
    )

    @classmethod
    def load_from_file(
        cls,
        filepath: Path,
        model_name: str,
        chunk_size: int,
    ) -> Manifest:
        manifest_default_data: dict = {
            "chunk_size": chunk_size,
            "model_name": model_name,
        }

        try:
            if filepath.exists():
                with open(filepath, "r") as f:
                    content = f.read()

                if not content:
                    return cls(**manifest_default_data)

                manifest_data = json.loads(content)

                if not isinstance(manifest_data, dict):
                    raise SchemaInvalidJSONRootError(
                        expected=dict, context=filepath
                    )

                manifest_data["already_exists"] = True
                manifest = cls(**{**manifest_default_data, **manifest_data})

                if (
                    manifest.chunk_size != chunk_size
                    or model_name != manifest.model_name
                ):
                    return cls(**manifest_default_data)

                return manifest
        except json.JSONDecodeError as e:
            raise SchemaInvalidJSONFormatError(
                context=filepath,
                lineno=e.lineno,
            ) from e

        return cls(**manifest_default_data)

    def sync(self, extensions: list[str]) -> set[str]:
        expired_chunk_ids: set[str] = set()

        self.extensions = set(extensions)

        if not self.already_exists:
            return expired_chunk_ids

        target_exts = set(extensions)
        current_exts = set(self.extensions).union(self.files_by_ext.keys())

        exts_to_delete = current_exts - target_exts
        exts_to_keep = current_exts.intersection(target_exts)

        for ext in exts_to_delete:
            if ext in self.files_by_ext:
                for cached_file in self.files_by_ext[ext].values():
                    expired_chunk_ids.update(cached_file.chunk_ids)
                del self.files_by_ext[ext]

        for ext in exts_to_keep:
            if ext not in self.files_by_ext:
                continue

            missing_file_ids = []
            for fileid, cached_file in self.files_by_ext[ext].items():
                if not Path(cached_file.filepath).exists():
                    expired_chunk_ids.update(cached_file.chunk_ids)
                    missing_file_ids.append(fileid)

            for fileid in missing_file_ids:
                del self.files_by_ext[ext][fileid]

            if not self.files_by_ext[ext]:
                del self.files_by_ext[ext]

        return expired_chunk_ids
