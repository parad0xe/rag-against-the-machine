from src.application.ports.loader import DocumentLoaderInterface
from src.domain.models.base import Chunk, Document, File, ManifestFileCache
from src.infrastructure.document.splitter import (
    LanguageTextSplitter,
)
from src.utils.common import md5


class DocumentLoader(DocumentLoaderInterface):
    def load(
        self,
        file: File,
        chunk_size: int,
        cached_file: ManifestFileCache | None = None,
    ) -> Document:
        str_file_path: str = str(file.path)

        raw_chunks = LanguageTextSplitter.from_filename(
            str_file_path,
            chunk_size=chunk_size,
        ).split_text(file.content)

        chunk_ids: list[str] = []
        chunk_metadatas: dict[str, Chunk] = {}

        search_index = 0
        for chunk in raw_chunks:
            chunk_start_index = file.content.find(chunk, search_index)
            if chunk_start_index == -1:
                chunk_start_index = search_index

            chunk_length = len(chunk)
            chunk_end_index = chunk_start_index + chunk_length
            chunk_id = f"chunk_{file.id}_{chunk_start_index}_{chunk_end_index}"
            chunk_hash = md5(chunk)

            chunk_ids.append(chunk_id)
            chunk_metadatas[chunk_id] = Chunk(
                text=chunk,
                hash=chunk_hash,
                file_path=str_file_path,
                first_character_index=chunk_start_index,
                last_character_index=chunk_end_index,
            )

            search_index = chunk_start_index + 1

        return Document(
            file=file,
            chunks=raw_chunks,
            chunk_ids=chunk_ids,
            chunk_metadatas=chunk_metadatas,
            stores=cached_file.stores if cached_file else set(),
        )
