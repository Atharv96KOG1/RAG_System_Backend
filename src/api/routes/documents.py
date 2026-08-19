import hashlib
import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.api.schemas.documents import (
    MAX_ACTIVE_DOCUMENTS,
    ActivateRequest,
    ActivateResponse,
    DocumentListResponse,
)
from src.api.state import state
from src.core.config import settings
from src.core.errors import RagError
from src.rag.document_parser import IMAGE_EXTENSIONS
from src.rag.pipeline import build_combined_chain, ingest_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf"} | IMAGE_EXTENSIONS


def _ingest_in_background(file_hash: str, dest_path):
    doc = state.documents[file_hash]
    try:
        entry = ingest_document(str(dest_path), state.embeddings)
    except RagError as exc:
        doc["ingest_status"] = "failed"
        doc["graph_status"] = "failed"
        doc["error"] = str(exc)
        return
    except Exception:
        logger.exception("Unhandled error ingesting %s", dest_path)
        doc["ingest_status"] = "failed"
        doc["graph_status"] = "failed"
        doc["error"] = "Something went wrong while processing this document."
        return

    doc["entry"] = entry
    doc["ingest_status"] = "ready"
    doc["graph_status"] = "ready" if entry.get("graph") is not None else "failed"


@router.get("", response_model=DocumentListResponse)
def list_documents():
    return DocumentListResponse(documents=state.document_list(), active_hashes=state.active_hashes)


@router.post("", response_model=DocumentListResponse)
def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF or image files (PNG/JPG/TIFF/BMP/WEBP) are supported.")

    file_bytes = file.file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()[:16]

    if file_hash not in state.documents:
        dest_path = settings.data_dir / file.filename
        dest_path.write_bytes(file_bytes)
        state.documents[file_hash] = {
            "source_path": dest_path,
            "filename": file.filename,
            "entry": None,
            "ingest_status": "pending",
            "graph_status": "pending",
            "error": None,
        }
        background_tasks.add_task(_ingest_in_background, file_hash, dest_path)

    return DocumentListResponse(documents=state.document_list(), active_hashes=state.active_hashes)


@router.get("/{file_hash}/file")
def get_document_file(file_hash: str):
    doc = state.documents.get(file_hash)
    if doc is None:
        raise HTTPException(status_code=404, detail="Unknown document hash.")

    media_type = mimetypes.guess_type(doc["filename"])[0] or "application/octet-stream"
    return FileResponse(
        doc["source_path"],
        media_type=media_type,
        filename=doc["filename"],
        content_disposition_type="inline",
    )


@router.get("/{file_hash}/pictures/{filename}")
def get_document_picture(file_hash: str, filename: str):
    if file_hash not in state.documents:
        raise HTTPException(status_code=404, detail="Unknown document hash.")

    path = settings.cache_dir / file_hash / "pictures" / Path(filename).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/png")


@router.delete("/{file_hash}", response_model=DocumentListResponse)
def remove_document(file_hash: str):
    state.documents.pop(file_hash, None)
    if file_hash in state.active_hashes:
        state.active_hashes = [h for h in state.active_hashes if h != file_hash]
        state.rag_chain = None
        state.doc_metadata = None
        state.combined_graph = None
        state.touched_box = {"node_ids": []}
        state.sources_box = {"items": []}
    return DocumentListResponse(documents=state.document_list(), active_hashes=state.active_hashes)


@router.post("/activate", response_model=ActivateResponse)
def activate_documents(request: ActivateRequest):
    if len(request.hashes) > MAX_ACTIVE_DOCUMENTS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_ACTIVE_DOCUMENTS} documents can be active.")

    missing = [h for h in request.hashes if h not in state.documents]
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown document hash(es): {', '.join(missing)}")

    not_ready = [h for h in request.hashes if state.documents[h]["ingest_status"] != "ready"]
    if not_ready:
        raise HTTPException(
            status_code=409, detail=f"Still processing document(s): {', '.join(not_ready)}. Try again shortly."
        )

    entries = [state.documents[h]["entry"] for h in request.hashes]
    try:
        rag_chain, doc_metadata, combined_graph, touched_box, sources_box = build_combined_chain(entries, state.device)
    except RagError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state.rag_chain = rag_chain
    state.doc_metadata = doc_metadata
    state.combined_graph = combined_graph
    state.touched_box = touched_box
    state.sources_box = sources_box
    state.active_hashes = request.hashes

    return ActivateResponse(active_hashes=state.active_hashes, combined_metadata=doc_metadata)
