from pydantic import ConfigDict

from src.domain.models.search import MinimalSearchResults


class MinimalAnswer(MinimalSearchResults):
    model_config = ConfigDict(extra="forbid")

    answer: str
