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

    metadata: DocumentMetadata | None
    ingest_status: str
    graph_status: str
    error: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    active_hashes: list[str]


class ActivateRequest(BaseModel):
    hashes: list[str] = Field(min_length=1, max_length=MAX_ACTIVE_DOCUMENTS)


class ActivateResponse(BaseModel):
    active_hashes: list[str]
    combined_metadata: dict
