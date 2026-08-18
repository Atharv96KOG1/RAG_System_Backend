import logging
from pathlib import Path

from src.core.config import cache_paths_for, settings
from src.rag.chunker import chunk_document, to_langchain_documents
from src.rag.document_parser import document_metadata, parse_document
from src.rag.embeddings import get_device, load_embeddings
from src.rag.graph_builder import build_combined_graph, build_document_graph, cache_graph, load_cached_graph
from src.rag.rag_chain import build_rag_chain, load_llm
from src.rag.retriever import build_graph_expanded_retriever, build_reranked_retriever, capture_sources
from src.rag.vector_store import build_vectorstore

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ModuleNotFoundError:
    from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)


def ingest_document(source_path, embeddings):
    """Parse -> chunk -> embed -> index -> graph-extract a single PDF. Cache and Milvus
    collection are keyed off the file's content hash (src.config.cache_paths_for), so
    ingesting a different file never reuses another document's parsed data, and
    ingesting the same file twice (e.g. re-selecting it) reuses the existing collection
    and graph instead of re-embedding/re-extracting. Returns everything needed to fold
    this doc into a combined retriever, but does not build a retriever/chain itself —
    call build_combined_chain for that."""
    paths = cache_paths_for(source_path)

    doc = parse_document(source_path, paths["document"])
    doc_metadata = document_metadata(doc)

    chunk_texts, chunk_metas = chunk_document(doc, paths["chunks"])
    lc_documents = to_langchain_documents(chunk_texts, chunk_metas)

    source_name = Path(source_path).name
    for lc_doc in lc_documents:
        lc_doc.metadata["source_file"] = source_name
        lc_doc.metadata["doc_hash"] = paths["key"]  # lets the frontend build a /api/documents/{hash}/file preview link

    vectorstore = build_vectorstore(lc_documents, embeddings, paths["collection_name"])

    graph = load_cached_graph(paths["graph"])
    if graph is None:
        try:
            llm = load_llm()
            graph = build_document_graph(chunk_texts, chunk_metas, source_name, llm)
            cache_graph(graph, paths["graph"])
        except Exception:
            # Graph extraction is additive (powers the graph-expanded retriever and the
            # Graph tab) — never let it fail the whole ingest. Chat still works off the
            # vector/BM25 retriever alone; the frontend just shows graph_status: "failed".
            logger.exception("Graph extraction failed for %s; continuing without it", source_path)
            graph = None

    return {
        "source_file": source_name,
        "cache_key": paths["key"],
        "doc_metadata": doc_metadata,
        "lc_documents": lc_documents,
        "vectorstore": vectorstore,
        "graph": graph,
    }


def _merge_doc_metadata(ingested_docs):
    totals = {"total_pages": 0, "total_tables": 0, "total_pictures": 0, "total_text_blocks": 0}
    for entry in ingested_docs:
        for key in totals:
            totals[key] += entry["doc_metadata"][key]
    totals["per_document"] = {entry["source_file"]: entry["doc_metadata"] for entry in ingested_docs}
    return totals


def build_combined_chain(ingested_docs, device):
    """Fold 1-4 already-ingested docs into one retriever/chain. Each doc keeps its own
    Milvus collection (built in ingest_document) — combining them for dense retrieval
    means one retriever per collection, ensembled together, rather than a single
    shared collection, so per-doc isolation (re-ingest doesn't disturb other docs)
    is preserved even when querying across several at once."""
    if not ingested_docs:
        raise ValueError("build_combined_chain requires at least one ingested document")

    lc_documents = [lc_doc for entry in ingested_docs for lc_doc in entry["lc_documents"]]

    dense_retrievers = [
        entry["vectorstore"].as_retriever(search_kwargs={"k": settings.dense_retriever_k}) for entry in ingested_docs
    ]
    bm25_retriever = BM25Retriever.from_documents(lc_documents)
    bm25_retriever.k = settings.bm25_retriever_k

    dense_weight, bm25_weight = settings.hybrid_weights
    weights = [dense_weight / len(dense_retrievers)] * len(dense_retrievers) + [bm25_weight]
    hybrid_retriever = EnsembleRetriever(retrievers=dense_retrievers + [bm25_retriever], weights=weights)

    combined_graph = build_combined_graph([entry.get("graph") for entry in ingested_docs])

    if combined_graph.number_of_nodes() > 0:
        retriever, touched_box = build_graph_expanded_retriever(
            hybrid_retriever, combined_graph, lc_documents, device, len(ingested_docs)
        )
    else:
        # No doc in this selection has a usable graph (all extraction failed, e.g. no
        # API key at ingest time) — degrade to the plain hybrid+rerank retriever rather
        # than erroring. Static empty box: nothing to highlight in the Graph tab.
        retriever = build_reranked_retriever(hybrid_retriever, device, len(ingested_docs))
        touched_box = {"node_ids": []}

    sources_box = {"items": []}
    retriever = capture_sources(retriever, sources_box)

    llm = load_llm()
    rag_chain = build_rag_chain(retriever, llm)

    return rag_chain, _merge_doc_metadata(ingested_docs), combined_graph, touched_box, sources_box


def build_pipeline(source_path):
    """Single-document convenience wrapper kept for the CLI: ingest one PDF and
    build a chain over it alone."""
    device = get_device()
    embeddings = load_embeddings(device)
    entry = ingest_document(source_path, embeddings)
    rag_chain, doc_metadata, combined_graph, _touched_box, _sources_box = build_combined_chain([entry], device)

    return {
        "rag_chain": rag_chain,
        "doc_metadata": doc_metadata,
        "cache_key": entry["cache_key"],
        "graph": combined_graph,
    }
