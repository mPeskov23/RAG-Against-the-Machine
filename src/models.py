"""Data models for RAG Against the Machine.

Provides Pydantic models for queries, retrieved sources, answers,
and evaluation metrics.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Union
import uuid
from pydantic import Field

if TYPE_CHECKING:
    class BaseModel:
        """Type checking stub for BaseModel."""

        def __init__(self, **kwargs: Any) -> None:
            ...

        def model_dump(self) -> Dict[str, Any]:
            ...

        def model_dump_json(self, indent: int = 2) -> str:
            ...

        @classmethod
        def model_validate(cls: Any, obj: Any) -> Any:
            ...
else:
    from pydantic import BaseModel


class MinimalSource(BaseModel):
    """Source location reference with file path and character spans."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question without ground-truth answers or retrieved sources."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with ground-truth sources and reference answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Collection of answered or unanswered questions."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Search results containing retrieved source locations for a question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search result bundled with model-generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Complete search results output for evaluation."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Complete search results and generated answers output for evaluation."""

    search_results: List[MinimalAnswer]
    k: int


class SearchRequest(BaseModel):
    """HTTP API search request body."""

    query: str
    k: int = 10
    hybrid: bool = False


class AnswerRequest(BaseModel):
    """HTTP API answer request body."""

    query: str
    k: int = 5
    hybrid: bool = False


class HealthResponse(BaseModel):
    """HTTP API health status response."""

    status: str
    indexed_documents: int
    lexical_index: bool
    semantic_index: bool
