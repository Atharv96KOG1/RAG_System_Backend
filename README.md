# Backend

FastAPI service wrapping the document-parsing → chunking → embedding → hybrid
retrieval → rerank → LLM pipeline. Also ships the original Streamlit app
(`src/views/app.py`) as an alternate UI.

## Structure

```
backend/
  main.py                 # entrypoint: `from src.api.server import app`
  src/
    api/
      server.py            # FastAPI app, CORS, router mounting
      state.py             # in-memory app state (ingested docs, active chain)
      routes/
        documents.py       # upload / list / activate / remove
        chat.py             # ask a question against the active documents
      schemas/              # pydantic request/response models
    controllers/
      pipeline.py           # ingest_document(), build_combined_chain(), build_pipeline()
    models/                 # document_parser, chunker, embeddings, vector_store,
                             # retriever, rag_chain, config, errors
    views/
      app.py                # original Streamlit UI (optional, still works)
  tests/
  data/                     # uploaded/sample PDFs
  cache/                    # per-document parse + chunk cache, keyed by content hash
```

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env   # set OPENAI_API_KEY, HF_TOKEN
uvicorn main:app --reload --port 8000
```

Requires a running Milvus instance at `http://localhost:19530` (see project root
for how it's provisioned).

## Test

```bash
pip install -r requirements-dev.txt
pytest
ruff check src/ tests/
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload a PDF (multipart `file`), ingest it (parse/chunk/embed/index) |
| `GET` | `/api/documents` | List ingested documents + currently active hashes |
| `DELETE` | `/api/documents/{hash}` | Remove a document from the in-memory list |
| `POST` | `/api/documents/activate` | `{hashes: string[]}` (max 4) — build a combined retriever/chain over them |
| `POST` | `/api/chat` | `{question: string}` — ask against the currently active documents |

Interactive docs at `http://localhost:8000/docs` once running.
