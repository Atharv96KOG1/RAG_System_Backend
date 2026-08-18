import logging
from pathlib import Path

from src.core.config import cache_paths_for, settings
from src.rag.chunker import chunk_document, to_langchain_documents
from src.rag.document_parser import document_metadata, parse_document
from src.rag.embeddings import get_device, load_embeddings
from src.rag.graph_builder import build_combined_graph, build_document_graph, cache_graph, load_cached_graph
from src.rag.rag_chain import build_rag_chain, load_llm
from src.rag.retriever import (
    build_graph_expanded_retriever,
    build_reranked_retriever,
    build_wide_table_retriever,
    capture_sources,
)
from src.rag.vector_store import build_vectorstore
from src.rag.vision import load_vision_llm

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ModuleNotFoundError:
    from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)


def ingest_document(source_path, embeddings):
    paths = cache_paths_for(source_path)
    source_name = Path(source_path).name

    logger.info("[%s] parsing document", source_name)
    doc = parse_document(source_path, paths["document"])
    doc_metadata = document_metadata(doc)

    logger.info("[%s] chunking document", source_name)
    chunk_texts, chunk_metas = chunk_document(doc, paths["chunks"])
    lc_documents = to_langchain_documents(chunk_texts, chunk_metas)

    for lc_doc in lc_documents:
        lc_doc.metadata["source_file"] = source_name
        lc_doc.metadata["doc_hash"] = paths["key"]

    logger.info("[%s] embedding %d chunks", source_name, len(lc_documents))
    vectorstore = build_vectorstore(lc_documents, embeddings, paths["collection_name"])

    graph = load_cached_graph(paths["graph"])
    if graph is None:
        try:
            logger.info("[%s] extracting graph", source_name)
            llm = load_llm()
            graph = build_document_graph(chunk_texts, chunk_metas, source_name, llm)
            cache_graph(graph, paths["graph"])
        except Exception:
            logger.exception("Graph extraction failed for %s; continuing without it", source_path)
            graph = None

    logger.info("[%s] ingest complete", source_name)
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
    if not ingested_docs:
        raise ValueError("build_combined_chain requires at least one ingested document")

    lc_documents = [lc_doc for entry in ingested_docs for lc_doc in entry["lc_documents"]]

    def make_hybrid_retriever(dense_k, bm25_k):
        dense_retrievers = [entry["vectorstore"].as_retriever(search_kwargs={"k": dense_k}) for entry in ingested_docs]
        bm25_retriever = BM25Retriever.from_documents(lc_documents)
        bm25_retriever.k = bm25_k
        dense_weight, bm25_weight = settings.hybrid_weights
        weights = [dense_weight / len(dense_retrievers)] * len(dense_retrievers) + [bm25_weight]
        return EnsembleRetriever(retrievers=dense_retrievers + [bm25_retriever], weights=weights)

    hybrid_retriever = make_hybrid_retriever(settings.dense_retriever_k, settings.bm25_retriever_k)

    combined_graph = build_combined_graph([entry.get("graph") for entry in ingested_docs])

    if combined_graph.number_of_nodes() > 0:
        retriever, touched_box, cross_encoder = build_graph_expanded_retriever(
            hybrid_retriever, combined_graph, lc_documents, device, len(ingested_docs)
        )
        graph_for_wide = combined_graph
    else:
        retriever, cross_encoder = build_reranked_retriever(hybrid_retriever, device, len(ingested_docs))
        touched_box = {"node_ids": []}
        graph_for_wide = None

    sources_box = {"items": []}
    retriever = capture_sources(retriever, sources_box)

    multiplier = settings.table_retrieval_breadth_multiplier
    wide_hybrid_retriever = make_hybrid_retriever(
        round(settings.dense_retriever_k * multiplier), round(settings.bm25_retriever_k * multiplier)
    )
    wide_retriever = build_wide_table_retriever(
        wide_hybrid_retriever, graph_for_wide, lc_documents, device, len(ingested_docs), cross_encoder
    )
    wide_retriever = capture_sources(wide_retriever, sources_box)

    llm = load_llm()
    vision_llm = load_vision_llm()
    rag_chain = build_rag_chain(retriever, llm, vision_llm, sources_box, wide_retriever=wide_retriever)

    return rag_chain, _merge_doc_metadata(ingested_docs), combined_graph, touched_box, sources_box


def build_pipeline(source_path):
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
