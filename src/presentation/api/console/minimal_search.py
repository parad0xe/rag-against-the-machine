from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.text import Text

from src.application.ports.display import SearchDisplayInterface


class MinimalSearchDisplayConsole(SearchDisplayInterface):
    def __init__(self) -> None:
        self._console = Console()

    def loader(self, message: str = "Searching...") -> Status:
        return self._console.status(
            f"[bold green]{message}[/]", spinner="dots2"
        )

    def title(self, query: str) -> None:
        self._console.print()
        self._console.print(
            Rule(
                title=f"[bold cyan]Search Results for: '{query}'[/]",
                style="cyan",
            )
        )

    def subquery(self, translated_query: str) -> None:
        self._console.print(
            f"[italic magenta]Translated to: '{translated_query}'[/]",
            justify="center",
        )

    def noresult(self) -> None:
        self._console.print("\n[bold red]No results found.[/]\n")

    def results(self, search_result) -> None:
        self._console.print(
            f"\n[bold]Question ID:[/] {search_result.question_id}\n"
        )

        for i, source in enumerate(search_result.retrieved_sources):
            content = Text()
            content.append("File     : ", style="bold magenta")
            content.append(f"{source.file_path}\n", style="bold white")
            content.append("Position : ", style="bold magenta")

            content.append(
                (
                    f"Chars {source.first_character_index} "
                    f"➔ {source.last_character_index}"
                ),
                style="yellow",
            )

            panel = Panel(
                content,
                title=f"[bold yellow]Rank {i + 1}[/]",
                title_align="left",
                border_style="blue",
                padding=(1, 2),
            )
            self._console.print(panel)

        self._console.print()
        self._console.print(
            Rule(
                title=(
                    f"[bold cyan]Total results: {len(search_result.retrieved_sources)}[/]"
                ),
                style="cyan",
            )
        )
        self._console.print()


# from rich.console import Console
# from rich.panel import Panel
# from rich.rule import Rule
# from rich.text import Text
#
# from src.domain.models.search import MinimalSearchResults
#
#
# class DisplayConsoleMinimalSearchResults:
#    def __init__(self) -> None:
#        self._console = Console()
#
#    def title(self, query: str) -> None:
#        self._console.print()
#
#        self._console.print(
#            Rule(
#                title=f"[bold cyan]🔎 Search Results for: '{query}'[/]",
#                style="cyan",
#            )
#        )
#
#    def subquery(self, translated_query: str) -> None:
#        self._console.print(
#            f"[italic magenta]Translated to: '{translated_query}'[/]",
#            justify="center",
#        )
#
#    def noresult(self) -> None:
#        self._console.print("[bold red]No results found.[/]\n")
#        pass
#
#    def results(self, search_result: MinimalSearchResults) -> None:
#        self._console.print(f"{search_result.question_id}")
#
#        for i, source in enumerate(search_result.retrieved_sources):
#            content = Text()
#            content.append("File     : ", style="bold magenta")
#            content.append(f"{source.file_path}\n", style="green")
#            content.append("Position : ", style="bold magenta")
#
#            content.append(
#                (
#                    f"Chars {source.first_character_index} "
#                    f"➔ {source.last_character_index}"
#                ),
#                style="yellow",
#            )
#
#            panel = Panel(
#                content,
#                title=f"[bold yellow]Rank {i + 1}[/]",
#                title_align="left",
#                border_style="blue",
#            )
#            self._console.print(panel)
#
#        self._console.print()
#        self._console.print(
#            Rule(
#                title=(
#                    "[bold cyan]Total results: "
#                    f"{len(search_result.retrieved_sources)}[/]"
#                ),
#                style="cyan",
#            )
#        )
#        self._console.print()
