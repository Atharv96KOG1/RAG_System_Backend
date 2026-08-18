import asyncio
import json
import re

import networkx as nx
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.core.config import settings

GRAPH_PROMPT = ChatPromptTemplate.from_template(
    """Extract a knowledge graph from the document section below. Only use entities and
relations explicitly stated in the text — never infer facts from outside knowledge, and
never invent an entity or relation that isn't grounded in this text.

Allowed entity types: {entity_types}. If an entity doesn't clearly fit one of these, use "other".

- Entities: distinct named or clearly-identifiable things (people, organizations, products,
  places, dates, key concepts/terms the section is actually about).
- Relations: a short verb/predicate phrase connecting two entities you already extracted
  (e.g. "works_for", "located_in", "reports_to", "part_of"). Every relation's source and
  target must exactly match a name in your entities list.
- Skip anything trivial or not worth graphing (e.g. don't extract every number or common word).

Section:
{text}
"""
)


class GraphEntity(BaseModel):
    name: str = Field(description="Entity's name exactly as it appears in the text.")
    type: str = Field(description="One of the allowed entity types.")


class GraphRelation(BaseModel):
    source: str = Field(description="Source entity name — must match an extracted entity's name.")
    target: str = Field(description="Target entity name — must match an extracted entity's name.")
    type: str = Field(description="Short relation label, e.g. 'works_for', 'part_of'.")


class ExtractedGraph(BaseModel):
    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)


def _approx_tokens(text):

    return len(text.split())


def group_chunks_by_heading(chunk_texts, chunk_metas, max_tokens=None):
    max_tokens = max_tokens if max_tokens is not None else settings.graph_group_max_tokens
    groups = []
    current_texts, current_indices, current_heading, current_tokens = [], [], object(), 0

    for i, (text, meta) in enumerate(zip(chunk_texts, chunk_metas, strict=True)):
        heading = meta.get("headings", "")
        tokens = _approx_tokens(text)
        starts_new_group = not current_indices or heading != current_heading or current_tokens + tokens > max_tokens
        if starts_new_group and current_indices:
            groups.append({"text": "\n\n".join(current_texts), "chunk_indices": current_indices})
            current_texts, current_indices, current_tokens = [], [], 0

        current_texts.append(text)
        current_indices.append(i)
        current_heading = heading
        current_tokens += tokens

    if current_indices:
        groups.append({"text": "\n\n".join(current_texts), "chunk_indices": current_indices})

    return groups


def _normalize_name(name):
    return re.sub(r"\s+", " ", name).strip().lower()


async def _extract_graph_for_group(group_text, llm, semaphore):
    structured_llm = llm.with_structured_output(ExtractedGraph)
    chain = GRAPH_PROMPT | structured_llm
    async with semaphore:
        try:
            return await chain.ainvoke({"entity_types": ", ".join(settings.graph_entity_types), "text": group_text})
        except Exception:
            return ExtractedGraph()


async def _extract_all_groups(groups, llm):
    semaphore = asyncio.Semaphore(settings.graph_extraction_concurrency)
    return await asyncio.gather(*(_extract_graph_for_group(g["text"], llm, semaphore) for g in groups))


def build_document_graph(chunk_texts, chunk_metas, source_file, llm):
    graph = nx.MultiDiGraph()

    for i, (text, meta) in enumerate(zip(chunk_texts, chunk_metas, strict=True)):
        graph.add_node(
            f"chunk:{source_file}:{i}",
            kind="chunk",
            source_file=source_file,
            page=meta.get("page"),
            content_type=meta.get("content_type"),
            preview=text[:160],
        )

    groups = group_chunks_by_heading(chunk_texts, chunk_metas)
    if not groups:
        return graph

    extracted = asyncio.run(_extract_all_groups(groups, llm))

    for group, result in zip(groups, extracted, strict=True):
        chunk_node_ids = [f"chunk:{source_file}:{i}" for i in group["chunk_indices"]]
        name_to_id = {}

        for entity in result.entities:
            node_id = _normalize_name(entity.name)
            if not node_id:
                continue
            name_to_id[_normalize_name(entity.name)] = node_id
            if graph.has_node(node_id):
                existing = graph.nodes[node_id]
                existing["source_files"] = sorted(set(existing["source_files"]) | {source_file})
                existing["chunk_ids"] = sorted(set(existing["chunk_ids"]) | set(chunk_node_ids))
            else:
                graph.add_node(
                    node_id,
                    kind="entity",
                    label=entity.name.strip(),
                    type=entity.type.strip().lower() or "other",
                    source_files=[source_file],
                    chunk_ids=list(chunk_node_ids),
                )
            for chunk_node_id in chunk_node_ids:
                graph.add_edge(chunk_node_id, node_id, type="MENTIONS")

        for relation in result.relations:
            source_id = name_to_id.get(_normalize_name(relation.source))
            target_id = name_to_id.get(_normalize_name(relation.target))
            if not source_id or not target_id or source_id == target_id:
                continue
            graph.add_edge(source_id, target_id, type=relation.type.strip() or "related_to", source_file=source_file)

    return graph


def cache_graph(graph, cache_path):
    cache_path.write_text(json.dumps(nx.node_link_data(graph)))


def load_cached_graph(cache_path):
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        return nx.node_link_graph(data, directed=True, multigraph=True)
    except (json.JSONDecodeError, KeyError):
        cache_path.unlink(missing_ok=True)
        return None


def build_combined_graph(per_doc_graphs):
    combined = nx.MultiDiGraph()
    for graph in per_doc_graphs:
        if graph is None:
            continue
        combined = nx.compose(combined, graph)

        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("kind") != "entity":
                continue
            merged = combined.nodes[node_id]
            merged["source_files"] = sorted(set(merged.get("source_files", [])) | set(attrs.get("source_files", [])))
            merged["chunk_ids"] = sorted(set(merged.get("chunk_ids", [])) | set(attrs.get("chunk_ids", [])))

    return combined


def overlap_node_ids(graph):
    return [
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("kind") == "entity"
        and attrs.get("type") in settings.graph_overlap_entity_types
        and len(attrs.get("source_files", [])) > 1
    ]
