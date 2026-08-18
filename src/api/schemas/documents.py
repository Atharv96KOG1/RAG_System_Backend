from pydantic import BaseModel, Field

MAX_ACTIVE_DOCUMENTS = 4


class DocumentMetadata(BaseModel):
    total_pages: int
    total_tables: int
    total_pictures: int
    total_text_blocks: int


class DocumentSummary(BaseModel):
    hash: str
    filename: str
    # None while ingest_status is "pending" — parsing/chunking/embedding hasn't finished yet.
    metadata: DocumentMetadata | None
    ingest_status: str  # "pending" | "ready" | "failed"
    graph_status: str  # "pending" | "ready" | "failed" — lags ingest_status, only gates the Graph tab
    error: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    active_hashes: list[str]


class ActivateRequest(BaseModel):
    hashes: list[str] = Field(min_length=1, max_length=MAX_ACTIVE_DOCUMENTS)


class ActivateResponse(BaseModel):
    active_hashes: list[str]
    combined_metadata: dict
