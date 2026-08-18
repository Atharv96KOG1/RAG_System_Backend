"""In-memory application state.

This is a single-process, single-deployment tool (one Milvus instance, one set of
locally-loaded models) — there is no multi-tenant isolation here, the same trade-off
the original Streamlit app made with st.session_state. A restart of the API process
clears active documents and the active chain; ingested documents themselves are safe
because ingest_document() persists them to cache/<hash>/ and their own Milvus
collection, so re-selecting a previously-ingested document after a restart is instant.
"""

import threading

from src.rag.embeddings import get_device, load_embeddings


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self._device = get_device()
        self._embeddings = None
        # file_hash -> {"source_path", "filename", "entry", "ingest_status", "graph_status", "error"}.
        # "entry" is None until ingestion (which runs in a background task, see
        # routes/documents.py) finishes; ingest_status/graph_status track that lifecycle
        # independently since a document is chat-usable as soon as ingest_status is
        # "ready" even if graph_status is still "pending" or ends up "failed".
        self.documents: dict[str, dict] = {}
        self.active_hashes: list[str] = []
        self.rag_chain = None
        self.doc_metadata: dict | None = None
        self.combined_graph = None
        # Mutable box shared by reference with the active graph-expanded retriever's
        # closure (see retriever.build_graph_expanded_retriever) — it overwrites
        # touched_box["node_ids"] on every /api/chat call, so this always reflects the
        # most recent query without routes/graph.py needing to reach into the chain.
        self.touched_box: dict = {"node_ids": []}
        # Same pattern as touched_box, but for chat citations — overwritten with the
        # exact chunks the retriever returned for the most recent /api/chat query.
        self.sources_box: dict = {"items": []}

    @property
    def embeddings(self):
        # Loaded lazily so importing this module (e.g. for tests) doesn't pull in
        # torch/sentence-transformers until a document is actually ingested.
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    self._embeddings = load_embeddings(self._device)
        return self._embeddings

    @property
    def device(self):
        return self._device

    def document_list(self):
        return [
            {
                "hash": file_hash,
                "filename": doc["filename"],
                "metadata": doc["entry"]["doc_metadata"] if doc["entry"] else None,
                "ingest_status": doc["ingest_status"],
                "graph_status": doc["graph_status"],
                "error": doc["error"],
            }
            for file_hash, doc in self.documents.items()
        ]

    def active_filenames(self):
        return [
            self.documents[h]["entry"]["source_file"]
            for h in self.active_hashes
            if h in self.documents and self.documents[h]["entry"]
        ]


state = AppState()
