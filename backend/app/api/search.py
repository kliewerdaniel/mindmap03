from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Dict
from ..services.embeddings import get_embedding_store
from ..services.graph_store import get_graph_store
from ..db.db import get_note

router = APIRouter()

class SemanticSearchRequest(BaseModel):
    q: str
    top_k: int = 10
    search_type: str = "both"  # "notes", "nodes", or "both"

class SearchResult(BaseModel):
    type: str  # "note" or "node"
    id: str
    content: str
    score: float
    metadata: Dict = {}

@router.post("/semantic")
async def semantic_search(request: SemanticSearchRequest):
    """
    Semantic search across notes and/or nodes.

    Args:
        q: Search query
        top_k: Number of results to return
        search_type: Search scope ("notes", "nodes", or "both")

    Returns:
        Ranked list of results
    """
    embedding_store = get_embedding_store()
    results = []

    if request.search_type in ["notes", "both"]:
        note_results = embedding_store.search_notes(request.q, request.top_k)
        for r in note_results:
            results.append(SearchResult(
                type="note",
                id=str(r['note_id']),
                content=r['content'][:200] + "..." if len(r['content']) > 200 else r['content'],
                score=r['score'],
                metadata=r['metadata']
            ))

    if request.search_type in ["nodes", "both"]:
        node_results = embedding_store.search_nodes(request.q, request.top_k)
        graph_store = get_graph_store()

        for r in node_results:
            node = graph_store.get_node(r['node_id'])
            if node:
                results.append(SearchResult(
                    type="node",
                    id=r['node_id'],
                    content=r['label'],
                    score=r['score'],
                    metadata={
                        'node_type': node.get('type'),
                        'provenance_count': len(node.get('provenance', []))
                    }
                ))

    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)

    # Limit to top_k
    results = results[:request.top_k]

    return {
        "query": request.q,
        "results": [r.dict() for r in results],
        "total": len(results)
    }

@router.get("/related/{node_id}")
async def get_related_nodes(
    node_id: str,
    top_k: int = Query(5, description="Number of related nodes to return")
):
    """
    Find semantically related nodes.

    Uses the node label as query to find similar nodes.
    """
    graph_store = get_graph_store()
    embedding_store = get_embedding_store()

    node = graph_store.get_node(node_id)
    if not node:
        return {"error": "Node not found"}

    # Search for similar nodes using label
    similar_nodes = embedding_store.search_nodes(node['label'], top_k + 1)

    # Filter out the query node itself
    similar_nodes = [n for n in similar_nodes if n['node_id'] != node_id][:top_k]

    return {
        "source_node": node_id,
        "related_nodes": similar_nodes
    }
