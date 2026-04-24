from pathlib import Path

from tqdm import tqdm

from src.domain.models.manifest import Manifest
from src.infrastructure.document.parser import DocumentParser
from src.infrastructure.indexers.stores.base import BaseIndexStore
from src.infrastructure.manifest.manager import ManifestManager
from src.utils.path_util import ensure_valide_dirpath, get_filepaths


class Indexer:
    def __init__(
        self,
        manifest_manager: ManifestManager,
        extensions: list[str],
        stores: list[BaseIndexStore],
    ) -> None:
        self._manifest_store: ManifestManager = manifest_manager
        self._stores: list[BaseIndexStore] = stores
        self._extensions: list[str] = extensions
        self._viewed_filepaths: set[Path] = set()

    def sync(self) -> None:
        manifest: Manifest = self._manifest_store.manifest
        expired_chunk_ids: set[str] = set()

        if self._extensions[0] == "*":
            target_exts = set(manifest.files_by_ext.keys())
            current_exts = set(manifest.files_by_ext.keys())
        else:
            target_exts = set(self._extensions)
            current_exts = set(self._extensions).union(
                manifest.files_by_ext.keys()
            )

        exts_to_delete = current_exts - target_exts
        exts_to_keep = current_exts.intersection(target_exts)

        for ext in exts_to_delete:
            if ext in manifest.files_by_ext:
                for cached_file in manifest.files_by_ext[ext].values():
                    for store in self._stores:
                        store.delete(cached_file.chunk_ids)
                del manifest.files_by_ext[ext]

        for ext in exts_to_keep:
            if ext not in manifest.files_by_ext:
                continue

            missing_file_ids = []
            for file_id, cached_file in manifest.files_by_ext[ext].items():
                if not Path(cached_file.file_path).exists():
                    for store in self._stores:
                        store.delete(cached_file.chunk_ids)
                    expired_chunk_ids.update(cached_file.chunk_ids)
                    missing_file_ids.append(file_id)
                if not any(
                    [
                        Path(repository).resolve()
                        in Path(cached_file.file_path).resolve().parents
                        for repository in manifest.repositories
                    ]
                ):
                    for store in self._stores:
                        store.delete(cached_file.chunk_ids)
                    expired_chunk_ids.update(cached_file.chunk_ids)
                    missing_file_ids.append(file_id)

            for file_id in missing_file_ids:
                del manifest.files_by_ext[ext][file_id]

            if not manifest.files_by_ext[ext]:
                del manifest.files_by_ext[ext]

    def index(self, repository: Path) -> None:
        ensure_valide_dirpath(repository)

        filepaths: list[str] = get_filepaths(
            repository,
            self._extensions,
            recursive=True,
        )
        document_parser = DocumentParser(self._manifest_store)
        for filepath in tqdm(filepaths, desc="Processing documents"):
            if Path(filepath).resolve() in self._viewed_filepaths:
                continue
            self._viewed_filepaths.add(Path(filepath).resolve())
            parse_result = document_parser.parse(Path(filepath))
            if not parse_result:
                continue

            document, status = parse_result

            if diff := self._manifest_store.diff(document):
                for store in self._stores:
                    store.delete(set(diff))

            for store in self._stores:
                store.add(document, status)
            self._manifest_store.update(document)

        for store in self._stores:
            print("TO DELETE:", len(store._delete_chunk_ids))
            print(
                "TO ADD:",
                sum([len(x.chunk_ids) for x in store._add_documents]),
            )
        print("manifest mismatch:", self._manifest_store.identity_mismatch)

    def commit(self) -> None:
        for store in self._stores:
            store.commit(self._manifest_store.identity_mismatch)
