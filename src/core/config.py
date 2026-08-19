import hashlib
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(str(BASE_DIR.parent / ".env"), str(BASE_DIR / ".env")), extra="ignore")
    openai_api_key: str | None = None

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    vision_model_name: str = "qwen/qwen3-vl-8b-instruct"
    milvus_uri: str = "http://localhost:19530"
    collection_prefix: str = "rag_doc_"
    data_dir: Path = BASE_DIR / "data"
    cache_dir: Path = BASE_DIR / "cache"
    source_path: Path = BASE_DIR / "data" / "26-004-crm-software-rfp-package.pdf"
    embed_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    reranker_max_length: int = 384
    llm_model_name: str = "gpt-4o-mini"
    chunk_max_tokens: int = 1024
    text_overlap_fraction: float = 0.15
    dense_retriever_k: int = 10
    bm25_retriever_k: int = 10
    hybrid_weights: list[float] = [0.6, 0.4]
    rerank_top_n: int = 8

    min_relevance_score: float = 0.15

    citation_relevance_fraction: float = 0.5
    graph_entity_types: list[str] = ["person", "organization", "product", "location", "concept", "date", "other"]
    graph_overlap_entity_types: list[str] = ["person", "organization", "product", "location"]
    graph_group_max_tokens: int = 2500
    graph_extraction_concurrency: int = 5
    graph_seed_k: int = 15
    graph_expand_limit: int = 10

    table_retrieval_breadth_multiplier: float = 3.0

    torch_thread_limit: int | None = None


settings = Settings()
settings.data_dir.mkdir(exist_ok=True)
settings.cache_dir.mkdir(exist_ok=True)


def cache_paths_for(source_path):
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
