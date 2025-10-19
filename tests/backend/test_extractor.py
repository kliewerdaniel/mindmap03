import pytest
from backend.app.services.extractor import (
    normalize_label,
    generate_node_id,
    parse_extraction_output
)
import json

def test_normalize_label():
    """Test label normalization."""
    assert normalize_label("Artificial Intelligence") == "artificial_intelligence"
    assert normalize_label("  AI  ") == "ai"
    assert normalize_label("Self-Driving Cars") == "self_driving_cars"

def test_generate_node_id():
    """Test deterministic node ID generation."""
    id1 = generate_node_id("test concept")
    id2 = generate_node_id("test concept")
    id3 = generate_node_id("different concept")

    assert id1 == id2  # Same label produces same ID
    assert id1 != id3  # Different labels produce different IDs
    assert id1.startswith("node:")

def test_parse_extraction_valid():
    """Test parsing valid extraction JSON."""
    valid_json = json.dumps({
        "nodes": [
            {"label": "sleep", "type": "concept", "span": [0, 5], "confidence": 0.9}
        ],
        "edges": [
            {"source": "sleep", "target": "health", "type": "affects", "confidence": 0.8}
        ],
        "summary": "Sleep affects health"
    })

    result = parse_extraction_output(valid_json)

    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["label"] == "sleep"
    assert len(result["edges"]) == 1
    assert result["summary"] == "Sleep affects health"

def test_parse_extraction_invalid_node_type():
    """Test parsing with invalid node type."""
    invalid_json = json.dumps({
        "nodes": [
            {"label": "test", "type": "invalid_type", "span": [0, 4], "confidence": 0.9}
        ],
        "edges": [],
        "summary": ""
    })

    with pytest.raises(ValueError, match="Invalid node type"):
        parse_extraction_output(invalid_json)

def test_parse_extraction_missing_fields():
    """Test parsing with missing required fields."""
    invalid_json = json.dumps({
        "nodes": [
            {"label": "test", "type": "concept"}  # Missing span and confidence
        ],
        "edges": []
    })

    with pytest.raises(ValueError, match="Invalid node structure"):
        parse_extraction_output(invalid_json)

def test_parse_extraction_with_extra_text():
    """Test parsing JSON embedded in text."""
    output_with_text = """
    Here is the extraction result:
    {"nodes": [{"label": "test", "type": "concept", "span": [0, 4], "confidence": 0.9}], "edges": [], "summary": "Test"}
    That's the analysis.
    """

    result = parse_extraction_output(output_with_text)

    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["label"] == "test"

# Mock LLM for integration testing
@pytest.fixture
def mock_llm_response(monkeypatch):
    """Mock LLM response for testing."""
    def mock_call_local_llm(prompt: str, model: str = None) -> str:
        return json.dumps({
            "nodes": [
                {"label": "sleep", "type": "concept", "span": [0, 5], "confidence": 0.95},
                {"label": "work", "type": "activity", "span": [20, 24], "confidence": 0.9}
            ],
            "edges": [
                {"source": "sleep", "target": "work", "type": "affects", "confidence": 0.9}
            ],
            "summary": "Sleep impacts work performance"
        })

    from backend.app.services import extractor
    monkeypatch.setattr(extractor, "call_local_llm", mock_call_local_llm)

def test_extract_from_text(mock_llm_response):
    """Test full extraction from text."""
    from backend.app.services.extractor import extract_from_text

    # Test with note_id directly
    result = extract_from_text("Sleep affects work", note_id=1)

    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1
    assert result["summary"] == "Sleep impacts work performance"
    assert all(node["note_id"] == 1 for node in result["nodes"])
