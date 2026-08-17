from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class SourceCitation(BaseModel):
    source_file: str | None = None
    doc_hash: str | None = None
    page: int | None = None
    content_type: str | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation] = []
