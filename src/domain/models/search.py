from pydantic import BaseModel, ConfigDict

from src.domain.models.source import MinimalSource


class MinimalSearchResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]
