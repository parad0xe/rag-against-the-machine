import logging
from pathlib import Path

from pydantic import (
    validate_call,
)
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from transformers import pipeline

from src.infrastructure.document.stores.bm25.query import BM25IndexStoreQuery
from src.infrastructure.document.stores.chroma.query import (
    ChromaIndexStoreQuery,
)
from src.infrastructure.document.stores.registry import IndexStoreQueryRegistry
from src.infrastructure.repositories.chunks import ChunksRepository
from src.infrastructure.retriever import Retriever
from src.utils.file import ensure_valid_dir_path

logger = logging.getLogger(__file__)


TRANSLATION_MODEL: str = "Helsinki-NLP/opus-mt-mul-en"


class Translator:
    def __init__(self) -> None:
        self.translator = pipeline(
            "translation",
            model=TRANSLATION_MODEL,
        )

    def translate_to_english(self, text: str) -> str:
        if not text.strip():
            return ""

        result = self.translator(
            text,
            max_length=512,
        )

        return str(result[0]["translation_text"])


@validate_call()
def entrypoint_search(
    query: str,
    bm25_dir_path: Path,
    chroma_dir_path: Path,
    chunks_file_path: Path,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    k: int = 10,
) -> None:
    console = Console()
    console.print()

    console.print(
        Rule(
            title=f"[bold cyan]🔎 Search Results for: '{query}'[/]",
            style="cyan",
        )
    )

    retriever = Retriever(
        index_store_registry=IndexStoreQueryRegistry(
            BM25IndexStoreQuery(bm25_dir_path, weight=0.65),
            ChromaIndexStoreQuery(
                chroma_dir_path,
                embedding_model_name,
                enable=True,  # faire en sorte de selectionner en fonction 'with semantic' dans le manifest
                weight=0.35,
            ),
        ),
        chunks_repository=ChunksRepository(chunks_file_path),
    )
    original_query = query
    query = Translator().translate_to_english(query)

    if original_query != query:
        console.print(
            f"[italic magenta]Translated to: '{query}'[/]", justify="center"
        )
    console.print()

    ensure_valid_dir_path(bm25_dir_path)
    ensure_valid_dir_path(chroma_dir_path)

    logger.info(f"Executing hybrid search for query: '{query}'")
    results = retriever.search(query, k=k)

    if not results:
        console.print("[bold red]No results found.[/]\n")
        return

    for i, source in enumerate(results):
        content = Text()
        content.append("File     : ", style="bold magenta")
        content.append(f"{source.file_path}\n", style="green")
        content.append("Position : ", style="bold magenta")

        # Regroupement des index sur une seule ligne avec une flèche
        content.append(
            f"Chars {source.first_character_index} ➔ {source.last_character_index}",
            style="yellow",
        )

        panel = Panel(
            content,
            title=f"[bold yellow]Rank {i + 1}[/]",
            title_align="left",
            border_style="blue",
        )
        console.print(panel)

    # Pied de page
    console.print()
    console.print(
        Rule(
            title=f"[bold cyan]Total results: {len(results)}[/]",
            style="cyan",
        )
    )
    console.print()
