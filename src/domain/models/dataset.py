import uuid

from pydantic import BaseModel, ConfigDict, Field


class MinimalSource(BaseModel):
    """
    Minimal representation of a document source.

    Attributes:
        file_path: Relative path to the source file.
        first_character_index: Start position in the source file.
        last_character_index: End position in the source file.
    """

    model_config = ConfigDict(extra="forbid")

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    Representation of a question without an associated answer.

    Attributes:
        question_id: Unique identifier for the question.
        question: The raw text of the question.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    Representation of a question with a ground truth answer and sources.

    Attributes:
        sources: List of minimal sources used to answer the question.
        answer: The ground truth answer text.
        difficulty: Categorization of the question difficulty.
        is_valid: Whether the question-answer pair is considered valid.
    """

    model_config = ConfigDict(extra="forbid")

    sources: list[MinimalSource]
    answer: str
    difficulty: str
    is_valid: bool


class RagDataset(BaseModel):
    """
    A collection of questions for RAG system evaluation.

    Attributes:
        rag_questions: List of answered or unanswered questions.
    """

    model_config = ConfigDict(extra="forbid")

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
