from pathlib import Path

from pydantic import validate_call
from rich import box, get_console
from rich.panel import Panel
from rich.table import Table

from src.application.services.manifest import ManifestService
from src.infrastructure.manifest.storage import ManifestJSONStorage


@validate_call()
def entrypoint_manifest_stats(manifest_file_path: Path) -> None:
    console = get_console()

    console.print()
    console.rule("[bold blue]Manifest statistics[/]", style="blue")
    console.print()

    service = ManifestService(storage=ManifestJSONStorage())
    stats = service.get_stats(manifest_file_path)

    semantic_color = (
        "[bold green]On[/]" if stats["with_semantic"] else "[bold red]Off[/]"
    )

    info = [
        f"[bold]Model:[/] {stats['embedding_model']}",
        f"[bold]Semantic:[/] {semantic_color}",
        f"[bold]Chunk size:[/] {stats['chunk_size']}",
        f"[bold]Fingerprint:[/] [dim]{stats['fingerprint']}[/]",
        "",
        f"[bold]Total files:[/] [yellow]{stats['total_files']}[/]",
        f"[bold]Total chunks:[/] [yellow]{stats['total_chunks']}[/]",
    ]

    config_panel = Panel.fit(
        "\n".join(info),
        title="[bold]Configuration[/]",
        border_style="blue",
        padding=(
            1,
            4,
        ),
    )
    console.print(config_panel)
    console.print()

    ext_table = Table(
        title="[bold dim]By extension[/]",
        header_style="bold magenta",
        border_style="cyan",
        box=box.ROUNDED,
        expand=False,
    )
    ext_table.add_column("Extension", style="bold cyan")
    ext_table.add_column("Files", justify="right", style="green")
    ext_table.add_column("Chunks", justify="right", style="yellow")

    for ext, s in stats["extensions"].items():
        ext_table.add_row(ext, str(s["file_count"]), str(s["chunk_count"]))

    console.print(ext_table)
    console.print()

    console.rule("[bold green]Manifest statistics completed[/]", style="blue")
    console.print()
