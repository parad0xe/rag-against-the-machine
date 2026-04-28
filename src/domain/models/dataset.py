import uuid

from pydantic import BaseModel, ConfigDict, Field


class MinimalSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    model_config = ConfigDict(extra="forbid")

    sources: list[MinimalSource]
    answer: str
    difficulty: str
    is_valid: bool


class RagDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
