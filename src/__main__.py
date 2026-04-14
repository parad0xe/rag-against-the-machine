import logging
import sys

import fire

from src.exceptions.base import RagError
from src.logger import LoggingSystem

logger = logging.getLogger(__file__)


class App:
    def __init__(self, verbose: int = 0, chunk_size: int = 2000) -> None:
        self._verbose: int = verbose
        self._chunk_size: int = chunk_size
        LoggingSystem.global_setup(verbose)

    def index(self) -> None:
        """
        Index the repository
        """
        raise NotImplementedError("App.index")

    def search(self) -> None:
        """
        Search for a single query
        """
        raise NotImplementedError("App.search")

    def search_dataset(self) -> None:
        """
        Process multiple questions and output search results
        """
        raise NotImplementedError("App.search_dataset")

    def answer(self) -> None:
        """
        Answer a single question with context
        """
        raise NotImplementedError("App.answer")


def main() -> None:
    try:
        fire.Fire(App)
    except RagError as e:
        print(
            f"[{e.__class__.__name__}] {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except BaseException as e:
        print(
            f"[{e.__class__.__name__}] An unexpected error occurred: {e}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
