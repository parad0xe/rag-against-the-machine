import json
from pathlib import Path
from typing import Literal, overload

from src.domain.exceptions.schema import (
    SchemaInvalidJSONFormatError,
    SchemaInvalidJSONRootError,
)
from src.domain.models.dataset import RagDataset
from src.utils.file import file_load_content


class RagDatasetJSONReader:
    @overload
    def read(
        self, file_path: Path, ignore_errors: Literal[False] = False
    ) -> RagDataset: ...

    @overload
    def read(
        self, file_path: Path, ignore_errors: Literal[True]
    ) -> RagDataset | None: ...

    @overload
    def read(
        self, file_path: Path, ignore_errors: bool
    ) -> RagDataset | None: ...

    def read(
        self,
        file_path: Path,
        ignore_errors: bool = False,
    ) -> RagDataset | None:
        content = file_load_content(
            file_path,
            ignore_errors=ignore_errors,
        )
        if not content:
            return None

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise SchemaInvalidJSONFormatError(
                context=file_path,
                lineno=e.lineno,
            ) from e

        if not isinstance(data, dict):
            if ignore_errors:
                return None
            raise SchemaInvalidJSONRootError(expected=dict, context=file_path)

        return RagDataset.model_validate(data)
