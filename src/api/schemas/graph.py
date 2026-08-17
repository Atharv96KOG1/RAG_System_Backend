from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    kind: str  # "entity" | "chunk"
    label: str
    type: str | None = None  # entity type (person/organization/...); None for chunk nodes
    source_files: list[str] = []
    page: int | None = None
    content_type: str | None = None
    preview: str | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    highlighted_node_ids: list[str]
    overlap_node_ids: list[str]
