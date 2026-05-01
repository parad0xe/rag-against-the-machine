import logging
import time
from pathlib import Path

from pydantic import PositiveInt, validate_call
from rich import get_console
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner

from src.factories.retriever import RetrieverFactory
from src.utils.format import build_context_from_chunks, parse_llm_thought

logger = logging.getLogger(__file__)


@validate_call()
def entrypoint_answer(
    original_query: str,
    bm25_dir_path: Path,
    chroma_dir_path: Path,
    chunks_file_path: Path,
    manifest_file_path: Path,
    embedding_model_name: str,
    k: PositiveInt,
    thinking: bool,
) -> None:
    console = get_console()

    console.print()
    console.rule("[bold blue]Answer[/]", style="blue")
    console.print(f"\n[bold]Query:[/] [cyan]{original_query}[/]\n")
    console.print()

    console.print("[bold cyan][1/3][/] Initializing environment")
    with console.status(
        "Loading models and parsing data",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        retriever, llm = RetrieverFactory.build(
            bm25_dir_path,
            chroma_dir_path,
            chunks_file_path,
            manifest_file_path,
            embedding_model_name,
        )

    console.print("[bold green][ OK ][/] Models and data loaded.\n")

    console.print("[bold cyan][2/3][/] Executing search query")
    with console.status(
        "Searching for relevant documents",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        _, chunks, translated_query = retriever.search(
            original_query=original_query,
            k=k,
        )
    console.print("[bold green][ OK ][/] Search completed.\n")

    console.print("[bold cyan][3/3][/] Generating answer")
    answer_stream = llm.generate_answer(
        query=translated_query,
        context=build_context_from_chunks(chunks),
        thinking=thinking,
    )

    full_text = ""
    start_time = time.perf_counter()

    thinking_spinner = Spinner("dots2", text="[cyan]Thinking...[/cyan]")

    with Live(
        refresh_per_second=15,
        console=console,
    ) as live:
        for token in answer_stream:
            full_text += token
            elapsed_time = time.perf_counter() - start_time
            time_str = f"[dim]Time elapsed: {elapsed_time:.1f}s[/dim]"

            thinking_text, final_text = parse_llm_thought(full_text)

            panels = []

            if thinking_text and thinking:
                panels.append(
                    Panel(
                        Markdown(thinking_text),
                        title="[dim]Thinking Process[/dim]",
                        title_align="left",
                        border_style="dim",
                        padding=(0, 2),
                    )
                )

            panels.append(
                Panel(
                    Markdown(final_text) if final_text else thinking_spinner,
                    title="[bold yellow]AI Answer[/]",
                    subtitle=time_str,
                    title_align="left",
                    subtitle_align="right",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

            live.update(Group(*panels))

    total_time = time.perf_counter() - start_time

    console.print(
        f"[bold green][ OK ][/] Answer generated in "
        f"[bold yellow]{total_time:.2f}s[/].\n"
    )
    console.rule("[bold green]Answer completed[/]", style="blue")
    console.print()
