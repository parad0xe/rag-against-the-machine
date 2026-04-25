from pathlib import Path

from src.domain.models.chunk import Chunk
from src.domain.models.document import Document
from src.infrastructure.document.splitter import LanguageTextSplitter
from src.utils.common import md5
from src.utils.file import file_load_content, get_extension


def load_document(
    file_path: Path, chunk_size: int, ignore_errors: bool = False
) -> Document | None:
    content = file_load_content(
        file_path,
        ignore_unicode_error=True,
        ignore_errors=ignore_errors,
    )
    if content is None:
        return None

    str_file_path = str(file_path)
    file_id = md5(str_file_path)
    file_hash = md5(content)
    ext = get_extension(str_file_path)

    raw_chunks = LanguageTextSplitter.from_filename(
        str_file_path,
        chunk_size=chunk_size,
        add_start_index=True,
    ).split_text(content)

    chunk_ids: list[str] = []
    chunk_metadatas: dict[str, Chunk] = {}

    start_index = 0
    for chunk in raw_chunks:
        chunk_length = len(chunk)
        chunk_start_index = start_index
        chunk_end_index = start_index + chunk_length
        chunk_id = f"chunk_{file_id}_{chunk_start_index}_{chunk_end_index}"
        chunk_hash = md5(chunk)

        chunk_ids.append(chunk_id)
        chunk_metadatas[chunk_id] = Chunk(
            text=chunk,
            hash=chunk_hash,
            file_path=str_file_path,
            first_character_index=chunk_start_index,
            last_character_index=chunk_end_index,
        )

        start_index += chunk_length

    return Document(
        id=file_id,
        hash=file_hash,
        ext=ext,
        file_path=str_file_path,
        content=content,
        chunks=raw_chunks,
        chunk_ids=chunk_ids,
        chunk_metadatas=chunk_metadatas,
    )
