"""Single source of truth for every model name, path, and tunable used across the
pipeline — chunking, embeddings, retrieval, GraphRAG extraction, and Milvus. Anything
that used to be a bare module-level constant now lives on the `settings` singleton
below, and every field can be overridden via env var or a .env file (pydantic-settings
matches env vars to field names case-insensitively) without touching code."""

import hashlib
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # The repo's real .env lives at the repo root (one level up from backend/), not
    # inside backend/ itself — listing both here (root first, backend/.env can still
    # override if it ever exists) instead of just BASE_DIR/".env" is what broke
    # OPENAI_API_KEY after the config.py move, since python-dotenv's old
    # load_dotenv() used to walk upward from the CWD and find it automatically.
    model_config = SettingsConfigDict(
        env_file=(str(BASE_DIR.parent / ".env"), str(BASE_DIR / ".env")), extra="ignore"
    )
    openai_api_key: str | None = None
    milvus_uri: str = "http://localhost:19530"
    collection_prefix: str = "rag_doc_"
    data_dir: Path = BASE_DIR / "data"
    cache_dir: Path = BASE_DIR / "cache"
    source_path: Path = BASE_DIR / "data" / "26-004-crm-software-rfp-package.pdf"
    embed_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    llm_model_name: str = "gpt-4o-mini"
    chunk_max_tokens: int = 1024
    text_overlap_fraction: float = 0.15
    dense_retriever_k: int = 10
    bm25_retriever_k: int = 10
    hybrid_weights: list[float] = [0.6, 0.4]
    rerank_top_n: int = 8
    graph_entity_types: list[str] = ["person", "organization", "product", "location", "concept", "date", "other"]
    graph_overlap_entity_types: list[str] = ["person", "organization", "product", "location"]
    graph_group_max_tokens: int = 2500
    graph_extraction_concurrency: int = 5
    graph_seed_k: int = 15
    graph_expand_limit: int = 10


settings = Settings()
settings.data_dir.mkdir(exist_ok=True)
settings.cache_dir.mkdir(exist_ok=True)


def cache_paths_for(source_path):
    """Cache keyed by file content hash, not a fixed filename — switching the source
    PDF (re-uploading a different file) can never silently reuse another doc's cached
    parse/chunks, which is what caused stale answers before this was per-file."""
    digest = hashlib.md5(Path(source_path).read_bytes()).hexdigest()[:16]
    doc_cache_dir = settings.cache_dir / digest
    doc_cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "key": digest,
        "document": doc_cache_dir / "document.cache.pkl",
        "chunks": doc_cache_dir / "chunks.cache.json",
        "graph": doc_cache_dir / "graph.cache.json",
        "collection_name": f"{settings.collection_prefix}{digest}",
    }
