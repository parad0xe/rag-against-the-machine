import logging
import time
from pathlib import Path

from pydantic import validate_call
from rich import get_console
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner

from src.infrastructure.factories.retriever import RetrieverFactory
from src.infrastructure.llm.qwen import QwenLlm

logger = logging.getLogger(__file__)

DEFAULT_MODEL_NAME: str = "Qwen/Qwen3-0.6B"


@validate_call()
def entrypoint_answer(
    original_query: str,
    bm25_dir_path: Path,
    chroma_dir_path: Path,
    chunks_file_path: Path,
    manifest_file_path: Path,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    k: int = 10,
    with_details: bool = False,
) -> None:
    console = get_console()

    console.print()
    console.rule("[bold blue]Answer[/]", style="blue")
    console.print(f"\n[bold]Query:[/] [cyan]{original_query}[/]\n")

    console.print("[bold cyan][1/3][/] Initializing environment...")
    with console.status(
        "Loading models and parsing data...",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        retriever, translator = RetrieverFactory.build(
            bm25_dir_path,
            chroma_dir_path,
            chunks_file_path,
            manifest_file_path,
            embedding_model_name,
        )

        llm = QwenLlm(model_name=DEFAULT_MODEL_NAME)
    console.print("[bold green][ OK ][/] Models and data loaded.\n")

    console.print("[bold cyan][2/3][/] Executing search query...")
    with console.status(
        "Searching for relevant documents...",
        spinner="dots",
        spinner_style="bold magenta",
    ):
        _, chunks = retriever.search(
            original_query=original_query,
            translator=translator,
            k=k,
        )
    console.print("[bold green][ OK ][/] Search completed.\n")

    context: list[str] = []
    for k, chunk in enumerate(chunks):
        file_path = chunk.get("file_path")
        start_index = chunk.get("first_character_index")
        end_index = chunk.get("last_character_index")
        text_content = chunk.get("text")

        context_source = (
            f"---  SOURCE #{k + 1} ---\n"
            f"File: {file_path} (Chars: {start_index}-{end_index})\n"
            f"Content: {text_content}"
        )
        context.append(context_source)

    console.print("[bold cyan][3/3][/] Generating answer...")
    answer_stream = llm.generate_answer(
        query=original_query,
        context="\n".join(context),
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

            thinking_text = ""
            final_text = ""

            if "<think>" in full_text:
                parts = full_text.split("<think>", 1)
                after_think = parts[1]
                if "</think>" in after_think:
                    think_parts = after_think.split("</think>", 1)
                    thinking_text = think_parts[0].strip("\n")
                    final_text = think_parts[1].strip("\n")
                else:
                    thinking_text = after_think.strip("\n")
            else:
                final_text = full_text.strip("\n")

            panels = []

            if thinking_text and with_details:
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
