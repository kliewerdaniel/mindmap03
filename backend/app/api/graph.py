from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel
from ..services.graph_store import get_graph_store
from pathlib import Path

router = APIRouter()

class NodeCreate(BaseModel):
    id: str
    label: str
    type: str
    provenance: List[List[int]] = []
    metadata: dict = {}

class EdgeCreate(BaseModel):
    source: str
    target: str
    type: str
    weight: float = 1.0
    extraction_id: Optional[int] = None

@router.get("/")
async def get_graph(
    node_id: Optional[str] = Query(None, description="Get subgraph around node"),
    depth: int = Query(2, description="Subgraph depth")
):
    """Get full graph or subgraph around a node."""
    graph_store = get_graph_store()

    if node_id:
        return graph_store.get_subgraph(node_id, depth)
    else:
        return {
            'nodes': graph_store.get_all_nodes(),
            'edges': graph_store.get_edges()
        }

@router.get("/node/{node_id}")
async def get_node(node_id: str):
    """Get specific node details."""
    graph_store = get_graph_store()
    node = graph_store.get_node(node_id)

    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    return node

@router.post("/node")
async def create_node(node: NodeCreate):
    """Create or update node."""
    graph_store = get_graph_store()

    node_id = graph_store.add_node(
        node.id,
        node.label,
        node.type,
        provenance=[tuple(p) for p in node.provenance],
        **node.metadata
    )

    graph_store.save()

    return {"node_id": node_id}

@router.post("/edge")
async def create_edge(edge: EdgeCreate):
    """Create or update edge."""
    graph_store = get_graph_store()

    try:
        graph_store.add_edge(
            edge.source,
            edge.target,
            edge.type,
            weight=edge.weight,
            extraction_id=edge.extraction_id
        )
        graph_store.save()
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/node/{node_id}")
async def delete_node(node_id: str):
    """Delete node and associated edges."""
    graph_store = get_graph_store()
    graph_store.delete_node(node_id)
    graph_store.save()
    return {"status": "deleted"}

@router.delete("/edge")
async def delete_edge(source: str, target: str):
    """Delete edge."""
    graph_store = get_graph_store()
    graph_store.delete_edge(source, target)
    graph_store.save()
    return {"status": "deleted"}

@router.get("/stats")
async def get_stats():
    """Get graph statistics."""
    graph_store = get_graph_store()
    return graph_store.get_stats()

@router.get("/export")
async def export_graph(format: str = Query("graphml", enum=["graphml", "gexf", "gpickle"])):
    """Export graph in specified format."""
    from fastapi.responses import FileResponse
    import tempfile

    graph_store = get_graph_store()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp:
        tmp_path = Path(tmp.name)

    if format == "graphml":
        graph_store.export_graphml(tmp_path)
    elif format == "gexf":
        graph_store.export_gexf(tmp_path)
    elif format == "gpickle":
        import shutil
        shutil.copy(graph_store.graph_path, tmp_path)

    return FileResponse(
        tmp_path,
        media_type="application/octet-stream",
        filename=f"mindmap_graph.{format}"
    )
