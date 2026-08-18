from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float


class SourceCitation(BaseModel):
    source_file: str | None = None
    doc_hash: str | None = None
    page: int | None = None
    content_type: str | None = None
    snippet: str
    relevance_score: float | None = None
    image_url: str | None = None
    bbox: BoundingBox | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation] = []
