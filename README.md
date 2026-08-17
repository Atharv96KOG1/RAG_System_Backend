# RAG System Backend

FastAPI service wrapping a GraphRAG pipeline: document parsing → chunking →
embedding → hybrid (vector + BM25) retrieval → graph-expanded retrieval →
rerank → LLM generation, with per-answer source citations and PDF page
preview. Also ships the original Streamlit app (`legacy/streamlit_app.py`)
as an alternate, superseded UI.

## Structure

```
main.py                    # entrypoint: `from src.api.server import app`
src/
  api/
    server.py              # FastAPI app, CORS, router mounting
    state.py                # in-memory app state (ingested docs, active chain/graph)
    routes/
      documents.py          # upload / list / activate / remove / file preview
      chat.py                # ask a question against the active documents
      graph.py                # entity/relation graph for visualization
    schemas/                 # pydantic request/response models
  core/
    config.py                # settings singleton — every model name, path, tunable
    errors.py                # RagError and subclasses
  rag/
    document_parser.py       # docling PDF parsing (OCR, tables, picture captions)
    chunker.py                 # chunking, sliding-window overlap, list/table detection
    embeddings.py               # BGE embedding model loading
    vector_store.py              # Milvus vector store
    retriever.py                  # hybrid + graph-expanded retrieval, reranking
    graph_builder.py               # per-document entity/relation extraction + graph
    rag_chain.py                    # prose/table LLM chains, citation formatting
    pipeline.py                      # ingest_document(), build_combined_chain()
  cli/
    chat_cli.py                # interactive terminal chat against one PDF
    parse_audit.py              # batch-parse every PDF in data/, report failures
legacy/
  streamlit_app.py            # original Streamlit UI, superseded by a separate frontend
tests/
scripts/
  eval_retrieval.py           # manual retrieval quality check (question -> expected page/keyword)
data/                          # uploaded/sample PDFs (gitignored)
cache/                         # per-document parse/chunk/graph cache, keyed by content hash (gitignored)
```

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY
uvicorn main:app --reload --port 8000
```

Requires a running Milvus instance at `http://localhost:19530`.

## Test

```bash
pip install -r requirements-dev.txt
pytest
ruff check src/ tests/ scripts/
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload a PDF (multipart `file`); ingestion (parse/chunk/embed/index/graph) runs in the background |
| `GET` | `/api/documents` | List ingested documents, their status, and currently active hashes |
| `GET` | `/api/documents/{hash}/file` | Serve the raw PDF inline, for page-level preview |
| `DELETE` | `/api/documents/{hash}` | Remove a document from the in-memory list |
| `POST` | `/api/documents/activate` | `{hashes: string[]}` (max 4) — build a combined retriever/chain over them |
| `POST` | `/api/chat` | `{question: string}` — ask against the active documents; returns the answer plus source citations |
| `GET` | `/api/graph` | Entity/relation graph for the active documents, with cross-document overlap flags |

Interactive docs at `http://localhost:8000/docs` once running.
