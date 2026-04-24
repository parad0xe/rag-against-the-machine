from pathlib import Path

from src.domain.models.document import ChunkMetadata, Document, DocumentStatus
from src.infrastructure.document.chain import CustomTextSplitter
from src.infrastructure.document.loader import DocumentLoader
from src.infrastructure.manifest.manager import ManifestManager
from src.utils.common_util import file_md5sum, md5sum
from src.utils.path_util import get_extension


class DocumentParser:
    def __init__(self, manifest_manager: ManifestManager) -> None:
        self._manifest_manager = manifest_manager
        self.reader = DocumentLoader()

    def parse(self, filepath: Path) -> tuple[Document, DocumentStatus] | None:
        content = self.reader.read(filepath)
        if not content:
            return None

        str_filepath = str(filepath)
        file_id = md5sum(str_filepath)
        file_hash = file_md5sum(str_filepath)
        ext = get_extension(str_filepath)

        raw_chunks = CustomTextSplitter.from_filename(
            str_filepath,
            chunk_size=self._manifest_manager.manifest.chunk_size,
            add_start_index=True,
        ).split_text(content)

        chunk_ids: list[str] = []
        chunk_metadatas: dict[str, ChunkMetadata] = {}

        start_index = 0
        for chunk in raw_chunks:
            chunk_length = len(chunk)
            chunk_start_index = start_index
            chunk_end_index = start_index + chunk_length
            chunk_id = f"chunk_{file_id}_{chunk_start_index}_{chunk_end_index}"
            chunk_hash = md5sum(chunk)

            chunk_ids.append(chunk_id)
            chunk_metadatas[chunk_id] = ChunkMetadata(
                text=chunk,
                hash=chunk_hash,
                file_path=str_filepath,
                first_character_index=chunk_start_index,
                last_character_index=chunk_end_index,
            )

            start_index += chunk_length

        document = Document(
            id=file_id,
            hash=file_hash,
            ext=ext,
            filepath=str_filepath,
            content=content,
            chunks=raw_chunks,
            chunk_ids=chunk_ids,
            chunk_metadatas=chunk_metadatas,
        )

        status = DocumentStatus.NEW
        cached_file = self._manifest_manager.get_cached_file(document)
        if cached_file:
            if self._manifest_manager.diff(document) is not None:
                status = DocumentStatus.UPDATE
            else:
                status = DocumentStatus.NOTHING_TO_DO

        return document, status
