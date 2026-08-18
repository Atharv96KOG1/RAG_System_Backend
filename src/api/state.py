import threading

from src.rag.embeddings import get_device, load_embeddings


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self._device = get_device()
        self._embeddings = None

        self.documents: dict[str, dict] = {}
        self.active_hashes: list[str] = []
        self.rag_chain = None
        self.doc_metadata: dict | None = None
        self.combined_graph = None

        self.touched_box: dict = {"node_ids": []}

        self.sources_box: dict = {"items": []}

    @property
    def embeddings(self):

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
