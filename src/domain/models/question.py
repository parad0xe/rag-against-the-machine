import uuid

from pydantic import BaseModel, ConfigDict, Field

from src.domain.models.source import MinimalSource


class UnansweredQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    model_config = ConfigDict(extra="forbid")

    sources: list[MinimalSource]
    answer: str
