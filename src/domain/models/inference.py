from pydantic import BaseModel, ConfigDict, Field

from src.domain.models.dataset import MinimalSource


class MinimalSearchResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    model_config = ConfigDict(extra="forbid")

    answer: str


class StudentSearchResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int
    search_results: list[MinimalSearchResults] = Field(default_factory=list)


class StudentSearchResultsAndAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int
    search_results: list[MinimalAnswer] = Field(default_factory=list)
