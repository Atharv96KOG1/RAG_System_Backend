from fastapi import APIRouter, HTTPException

from src.api.schemas.chat import ChatRequest, ChatResponse, SourceCitation
from src.api.state import state
from src.rag.rag_chain import answer_query

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    if state.rag_chain is None:
        raise HTTPException(status_code=400, detail="No active documents. Activate at least one document first.")

    # Reset first so a failed/short-circuited call (e.g. rate limit, or a meta question
    # like "how many pages" that never touches the retriever) reports no sources rather
    # than stale citations left over from a previous question.
    state.sources_box["items"] = []

    # answer_query already catches rate-limit/timeout/API errors from the LLM call
    # and returns a friendly string instead of raising, so no try/except needed here.
    answer = answer_query(request.question, state.rag_chain, state.doc_metadata)
    sources = [SourceCitation(**item) for item in state.sources_box.get("items", [])]
    return ChatResponse(answer=answer, sources=sources)
