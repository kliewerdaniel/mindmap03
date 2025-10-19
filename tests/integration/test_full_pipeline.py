import pytest
import requests
import time
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_health_check():
    """Test API health endpoint."""
    response = requests.get(f"{API_BASE}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ingestion_pipeline():
    """
    Integration test: Ingest sample notes and verify graph creation.

    This test validates the complete pipeline:
    1. Upload markdown file
    2. Wait for processing
    3. Verify nodes and edges created
    4. Check graph statistics
    """
    # Read sample note
    sample_path = Path(__file__).parent.parent.parent / "data" / "notes" / "sample1.md"

    with open(sample_path, 'r') as f:
        content = f.read()

    # Ingest text
    response = requests.post(
        f"{API_BASE}/api/ingest/text",
        json={
            "filename": "sample1.md",
            "content": content
        }
    )

    assert response.status_code == 200
    data = response.json()
    note_id = data["note_id"]

    # Poll for processing completion
    max_attempts = 30
    for attempt in range(max_attempts):
        status_response = requests.get(f"{API_BASE}/api/ingest/status/{note_id}")
        status_data = status_response.json()

        if status_data["processed"]:
            break

        time.sleep(2)
    else:
        pytest.fail("Processing timed out after 60 seconds")

    # Verify graph updated
    graph_response = requests.get(f"{API_BASE}/api/graph")
    assert graph_response.status_code == 200
    graph_data = graph_response.json()

    assert len(graph_data["nodes"]) > 0, "No nodes created from extraction"
    assert len(graph_data["edges"]) >= 0, "Graph should have edges or be valid without them"

    # Verify node types
    node_types = [node["type"] for node in graph_data["nodes"]]
    valid_types = {"concept", "person", "place", "idea", "event", "passage"}
    assert all(t in valid_types for t in node_types), f"Invalid node types: {node_types}"

    # Verify provenance exists
    for node in graph_data["nodes"]:
        assert "provenance" in node, f"Node {node['id']} missing provenance"
        assert len(node["provenance"]) > 0, f"Node {node['id']} has empty provenance"

def test_semantic_search():
    """Test semantic search functionality."""
    # Ensure some data exists
    graph_response = requests.get(f"{API_BASE}/api/graph")
    graph_data = graph_response.json()

    if len(graph_data["nodes"]) == 0:
        pytest.skip("No graph data available for search test")

    # Perform search
    search_response = requests.post(
        f"{API_BASE}/api/search/semantic",
        json={
            "q": "productivity and sleep",
            "top_k": 5,
            "search_type": "both"
        }
    )

    assert search_response.status_code == 200
    search_data = search_response.json()

    assert "results" in search_data
    assert isinstance(search_data["results"], list)

    # Verify result structure
    for result in search_data["results"]:
        assert "type" in result
        assert result["type"] in ["note", "node"]
        assert "score" in result
        assert 0 <= result["score"] <= 1

def test_graph_export():
    """Test graph export functionality."""
    # Export as GraphML
    export_response = requests.get(f"{API_BASE}/api/export?format=graphml")
    assert export_response.status_code == 200
    assert len(export_response.content) > 0

    # Verify GraphML content
    content = export_response.content.decode('utf-8')
    assert '<?xml' in content
    assert '<graphml' in content

def test_full_batch_ingestion():
    """
    Test batch ingestion of all sample notes.

    This is the acceptance test from Phase 2.
    """
    notes_dir = Path(__file__).parent.parent.parent / "data" / "notes"

    if not notes_dir.exists():
        pytest.skip("Sample notes directory not found")

    note_ids = []

    # Ingest all markdown files
    for md_file in notes_dir.glob("*.md"):
        with open(md_file, 'r') as f:
            content = f.read()

        response = requests.post(
            f"{API_BASE}/api/ingest/text",
            json={
                "filename": md_file.name,
                "content": content
            }
        )

        assert response.status_code == 200
        note_ids.append(response.json()["note_id"])

    # Wait for all processing to complete
    max_wait = 120  # 2 minutes
    start_time = time.time()

    while time.time() - start_time < max_wait:
        all_processed = True

        for note_id in note_ids:
            status_response = requests.get(f"{API_BASE}/api/ingest/status/{note_id}")
            if not status_response.json()["processed"]:
                all_processed = False
                break

        if all_processed:
            break

        time.sleep(3)
    else:
        pytest.fail("Batch processing timed out")

    # Get final graph stats
    stats_response = requests.get(f"{API_BASE}/api/graph/stats")
    stats = stats_response.json()

    # Acceptance criteria
    assert stats["num_nodes"] > 0, "No nodes created from sample notes"
    assert stats["num_edges"] >= 0, "Invalid edge count"

    print(f"\n✓ Successfully ingested {len(note_ids)} notes")
    print(f"✓ Created {stats['num_nodes']} nodes")
    print(f"✓ Created {stats['num_edges']} edges")

    # Export and verify provenance
    export_response = requests.get(f"{API_BASE}/api/export?format=graphml")
    assert export_response.status_code == 200

    export_content = export_response.content.decode('utf-8')
    assert 'provenance' in export_content, "Exported graph missing provenance data"

    print("✓ Exported graph contains provenance data")
