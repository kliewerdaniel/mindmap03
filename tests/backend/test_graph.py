import pytest
from backend.app.services.graph_store import GraphStore
from pathlib import Path
import tempfile

@pytest.fixture
def temp_graph():
    """Create temporary graph for testing."""
    with tempfile.NamedTemporaryFile(suffix=".gpickle", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    graph_store = GraphStore(tmp_path)

    yield graph_store

    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()

def test_add_node(temp_graph):
    """Test node addition."""
    node_id = temp_graph.add_node(
        "node:1",
        "Test Node",
        "concept",
        provenance=[(1, 0, 10)]
    )

    assert node_id == "node:1"
    assert temp_graph.graph.has_node("node:1")

    node = temp_graph.get_node("node:1")
    assert node['label'] == "Test Node"
    assert node['type'] == "concept"
    assert len(node['provenance']) == 1

def test_add_edge(temp_graph):
    """Test edge addition."""
    temp_graph.add_node("node:1", "Node 1", "concept")
    temp_graph.add_node("node:2", "Node 2", "concept")

    temp_graph.add_edge("node:1", "node:2", "related_to", weight=0.9)

    assert temp_graph.graph.has_edge("node:1", "node:2")

    edges = temp_graph.get_edges("node:1")
    assert len(edges) == 1
    assert edges[0]['type'] == "related_to"
    assert edges[0]['weight'] == 0.9

def test_persistence(temp_graph):
    """Test graph save and load."""
    temp_graph.add_node("node:1", "Test Node", "concept")
    temp_graph.add_node("node:2", "Test Node 2", "person")
    temp_graph.add_edge("node:1", "node:2", "related_to")

    temp_graph.save()

    # Create new instance with same path
    new_graph = GraphStore(temp_graph.graph_path)

    assert new_graph.graph.has_node("node:1")
    assert new_graph.graph.has_node("node:2")
    assert new_graph.graph.has_edge("node:1", "node:2")

def test_merge_provenance(temp_graph):
    """Test provenance merging on node update."""
    temp_graph.add_node("node:1", "Test", "concept", provenance=[(1, 0, 10)])
    temp_graph.add_node("node:1", "Test", "concept", provenance=[(2, 5, 15)])

    node = temp_graph.get_node("node:1")
    assert len(node['provenance']) == 2
    assert (1, 0, 10) in node['provenance']
    assert (2, 5, 15) in node['provenance']

def test_get_neighbors(temp_graph):
    """Test neighbor retrieval."""
    temp_graph.add_node("node:1", "Node 1", "concept")
    temp_graph.add_node("node:2", "Node 2", "concept")
    temp_graph.add_node("node:3", "Node 3", "concept")

    temp_graph.add_edge("node:1", "node:2", "related_to")
    temp_graph.add_edge("node:2", "node:3", "related_to")

    neighbors_d1 = temp_graph.get_neighbors("node:1", depth=1)
    assert "node:2" in neighbors_d1
    assert "node:3" not in neighbors_d1

    neighbors_d2 = temp_graph.get_neighbors("node:1", depth=2)
    assert "node:2" in neighbors_d2
    assert "node:3" in neighbors_d2

def test_subgraph(temp_graph):
    """Test subgraph extraction."""
    temp_graph.add_node("node:1", "Node 1", "concept")
    temp_graph.add_node("node:2", "Node 2", "concept")
    temp_graph.add_node("node:3", "Node 3", "concept")
    temp_graph.add_node("node:4", "Node 4", "concept")

    temp_graph.add_edge("node:1", "node:2", "related_to")
    temp_graph.add_edge("node:2", "node:3", "related_to")
    temp_graph.add_edge("node:3", "node:4", "related_to")

    subgraph = temp_graph.get_subgraph("node:2", depth=1)

    node_ids = [n['id'] for n in subgraph['nodes']]
    assert "node:2" in node_ids
    assert "node:1" in node_ids
    assert "node:3" in node_ids
    assert "node:4" not in node_ids

def test_centrality(temp_graph):
    """Test centrality computation."""
    temp_graph.add_node("node:1", "Node 1", "concept")
    temp_graph.add_node("node:2", "Node 2", "concept")
    temp_graph.add_node("node:3", "Node 3", "concept")

    temp_graph.add_edge("node:1", "node:2", "related_to")
    temp_graph.add_edge("node:1", "node:3", "related_to")
    temp_graph.add_edge("node:2", "node:3", "related_to")

    centrality = temp_graph.compute_centrality("degree")

    assert "node:1" in centrality
    assert "node:2" in centrality
    assert "node:3" in centrality
    assert centrality["node:1"] > 0
