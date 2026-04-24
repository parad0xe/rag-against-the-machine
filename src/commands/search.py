import logging
from pathlib import Path

import bm25s
from pydantic import (
    validate_call,
)

from src.models.chunk import ChunkMetadata
from src.models.stats import Stats
from src.utils.path_util import (
    ensure_valide_dirpath,
    ensure_valide_filepath,
)

logger = logging.getLogger(__file__)


@validate_call()
def entrypoint_search(
    query: str,
    k: int = 10,
    index_dirpath: Path = Path("data/processed/bm25_index"),
    stats_filepath: Path = Path("data/processed/stats.dat"),
) -> None:
    ensure_valide_filepath(stats_filepath)
    ensure_valide_dirpath(index_dirpath)

    stats = Stats.load(stats_filepath)
    print(stats)

    reloaded_retriever = bm25s.BM25.load(index_dirpath, load_corpus=True)

    query_tokens = bm25s.tokenize(query)
    results, scores = reloaded_retriever.retrieve(
        query_tokens,
        k=min(max(0, k), stats.num_chunks),
        n_threads=8,
    )

    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        chunk_metadata = ChunkMetadata(**doc)
        del chunk_metadata.text
        print(chunk_metadata, score)

    raise NotImplementedError("App.search")
