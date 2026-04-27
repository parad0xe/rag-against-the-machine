from pydantic import BaseModel, ConfigDict

from src.domain.models.question import AnsweredQuestion, UnansweredQuestion


class RagDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
