from fastapi import APIRouter, HTTPException

from src.api.schemas.graph import GraphEdge, GraphNode, GraphResponse
from src.api.state import state
from src.rag.graph_builder import overlap_node_ids

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("", response_model=GraphResponse)
def get_graph():
    graph = state.combined_graph
    if graph is None or graph.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="No active documents with a usable graph.")

    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("kind") == "entity":
            nodes.append(
                GraphNode(
                    id=node_id,
                    kind="entity",
                    label=attrs.get("label", node_id),
                    type=attrs.get("type"),
                    source_files=attrs.get("source_files", []),
                )
            )
        elif graph.degree(node_id) > 0:
            nodes.append(
                GraphNode(
                    id=node_id,
                    kind="chunk",
                    label=attrs.get("preview", node_id),
                    source_files=[attrs["source_file"]] if attrs.get("source_file") else [],
                    page=attrs.get("page"),
                    content_type=attrs.get("content_type"),
                    preview=attrs.get("preview"),
                )
            )

    kept_ids = {n.id for n in nodes}
    edges = [
        GraphEdge(source=u, target=v, type=attrs.get("type", "related_to"))
        for u, v, attrs in graph.edges(data=True)
        if u in kept_ids and v in kept_ids
    ]

    return GraphResponse(
        nodes=nodes,
        edges=edges,
        highlighted_node_ids=state.touched_box.get("node_ids", []),
        overlap_node_ids=overlap_node_ids(graph),
    )
