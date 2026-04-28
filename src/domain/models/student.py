from pydantic import BaseModel, ConfigDict, Field

from src.domain.models.answer import MinimalAnswer
from src.domain.models.search import MinimalSearchResults


class StudentSearchResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int
    search_results: list[MinimalSearchResults] = Field(default_factory=list)


class StudentSearchResultsAndAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int
    search_results: list[MinimalAnswer] = Field(default_factory=list)
