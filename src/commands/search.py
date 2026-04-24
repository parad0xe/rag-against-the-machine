import json
import logging
from pathlib import Path

import bm25s
import chromadb
from chromadb.config import Settings
from pydantic import (
    validate_call,
)
from sentence_transformers import SentenceTransformer

from src.utils.common_util import compute_rrf
from src.utils.path_util import ensure_valid_dirpath

logger = logging.getLogger(__file__)


@validate_call()
def entrypoint_search(
    query: str,
    k: int = 10,
    bm25_dirpath: Path = Path("data/processed/bm25_index"),
    chroma_dirpath: Path = Path("data/processed/chroma_index"),
    chunks_filepath: Path = Path("data/processed/chunks.json"),
    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
) -> None:
    Retriever(stores=[BM25Store(), ChromaStore()])
    ensure_valid_dirpath(bm25_dirpath)
    ensure_valid_dirpath(chroma_dirpath)

    # --- 1. RECHERCHE BM25 (Mots-clés) ---
    logger.info(f"Querying BM25 index from '{bm25_dirpath}'")
    bm25_retriever = bm25s.BM25.load(bm25_dirpath, load_corpus=True)

    corpus_size = len(bm25_retriever.corpus) if bm25_retriever.corpus else 0
    actual_k = min(k, corpus_size)

    bm25_ids: list[str] = []
    if actual_k > 0:
        query_tokens = bm25s.tokenize(query)
        bm25_results, _ = bm25_retriever.retrieve(
            query_tokens, k=actual_k, n_threads=1
        )
        if bm25_results.shape[1] > 0:
            bm25_ids = [doc.get("id") for doc in bm25_results[0]]

    # --- 2. RECHERCHE CHROMA (Sémantique) ---
    logger.info(f"Querying ChromaDB index from '{chroma_dirpath}'")
    client = chromadb.PersistentClient(
        path=str(chroma_dirpath),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name="chunks")

    model = SentenceTransformer(embedding_model_name)
    query_embedding = model.encode(query, convert_to_numpy=True)

    chroma_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=actual_k,
    )
    chroma_ids: list[str] = (
        chroma_results["ids"][0] if chroma_results["ids"] else []
    )

    # --- 3. FUSION RRF ---
    rrf_results = compute_rrf(bm25_ids, chroma_ids)
    top_results = rrf_results[:k]

    # --- 4. AFFICHAGE ---
    chunks_data = {}
    if chunks_filepath.exists():
        with open(chunks_filepath, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

    print(
        f"\n{'=' * 60}\n🔎 Hybrid Search Results for: '{query}'\n{'=' * 60}\n"
    )

    for i, (chunk_id, score) in enumerate(top_results):
        print(f"🥇 Rank {i + 1} | RRF Score: {score:.4f} | ID: {chunk_id}")

        if chunk_id in chunks_data:
            chunk_meta = chunks_data[chunk_id]
            file_path = chunk_meta.get("file_path", "Unknown")
            text = chunk_meta.get("text", "").strip()

            snippet = text

            print(f"📄 File: {file_path}")
            print(f"💬 Content:\n{snippet}\n")
        else:
            print("⚠️ Content: <No data found in chunks.json>\n")

    print("=" * 60)
