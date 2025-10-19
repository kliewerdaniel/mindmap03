import requests
import json
from typing import Dict, List, Tuple, Optional
from ..config import settings
from ..db.db import insert_extract, mark_note_processed, get_note
from .graph_store import get_graph_store
import hashlib
import re

EXTRACTION_PROMPT_TEMPLATE = """You are a knowledge extraction assistant. Analyze the following text and extract structured information in strict JSON format.

Required JSON Schema:
{{
  "nodes": [
    {{"label": "string", "type": "concept|person|place|idea|event|passage", "span": [start, end], "confidence": 0.0-1.0}}
  ],
  "edges": [
    {{"source": "label", "target": "label", "type": "related_to|causes|elaborates|contradicts|similar_to|part_of|precedes|affects", "confidence": 0.0-1.0}}
  ],
  "summary": "one-sentence summary"
}}

Edge types:
- related_to: General association
- causes: Causal relationship
- elaborates: Provides detail
- contradicts: Conflicting information
- similar_to: Conceptual similarity
- part_of: Hierarchical relationship
- precedes: Temporal ordering
- affects: Impact or influence

Return ONLY valid JSON. No additional text.

Text to analyze:
\"\"\"
{text}
\"\"\"
"""

def normalize_label(label: str) -> str:
    """Normalize entity label for consistent node IDs."""
    # Lowercase, remove special chars except hyphens, replace spaces and hyphens with underscores
    normalized = re.sub(r'[^\w\s-]', '', label.lower())
    normalized = re.sub(r'[-\s]+', '_', normalized)
    return normalized.strip('_')

def generate_node_id(label: str) -> str:
    """Generate unique node ID from label."""
    normalized = normalize_label(label)
    # Use hash for uniqueness while keeping it deterministic
    hash_suffix = hashlib.md5(normalized.encode()).hexdigest()[:8]
    return f"node:{normalized}_{hash_suffix}"

def call_local_llm(prompt: str, model: str = None) -> str:
    """
    Call local LLM endpoint (Ollama format).

    Args:
        prompt: The prompt text
        model: Model name (defaults to settings.llm_model)

    Returns:
        Generated text response

    Raises:
        Exception: If LLM call fails
    """
    model = model or settings.llm_model

    try:
        response = requests.post(
            settings.llm_endpoint,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent extraction
                    "num_predict": 2048
                }
            },
            timeout=settings.extraction_timeout
        )
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    except requests.exceptions.Timeout:
        raise Exception("LLM request timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"LLM request failed: {str(e)}")

def parse_extraction_output(llm_output: str) -> Dict:
    """
    Parse and validate LLM extraction output.

    Args:
        llm_output: Raw LLM response string

    Returns:
        Parsed and validated extraction dict

    Raises:
        ValueError: If output is invalid JSON or missing required fields
    """
    # Try to extract JSON from output (handle cases where LLM adds extra text)
    json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in LLM output")

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")

    # Validate schema
    if "nodes" not in data or not isinstance(data["nodes"], list):
        raise ValueError("Missing or invalid 'nodes' field")

    if "edges" not in data or not isinstance(data["edges"], list):
        raise ValueError("Missing or invalid 'edges' field")

    if "summary" not in data:
        data["summary"] = ""  # Optional field

    # Validate node structure
    valid_node_types = {"concept", "person", "place", "idea", "event", "passage"}
    for node in data["nodes"]:
        if not all(k in node for k in ["label", "type", "span", "confidence"]):
            raise ValueError(f"Invalid node structure: {node}")

        if node["type"] not in valid_node_types:
            raise ValueError(f"Invalid node type: {node['type']}")

        if not isinstance(node["span"], list) or len(node["span"]) != 2:
            raise ValueError(f"Invalid span format: {node['span']}")

        if not 0 <= node["confidence"] <= 1:
            raise ValueError(f"Invalid confidence score: {node['confidence']}")

    # Validate edge structure
    valid_edge_types = {
        "related_to", "causes", "elaborates", "contradicts",
        "similar_to", "part_of", "precedes", "affects"
    }
    for edge in data["edges"]:
        if not all(k in edge for k in ["source", "target", "type", "confidence"]):
            raise ValueError(f"Invalid edge structure: {edge}")

        if edge["type"] not in valid_edge_types:
            raise ValueError(f"Invalid edge type: {edge['type']}")

        if not 0 <= edge["confidence"] <= 1:
            raise ValueError(f"Invalid confidence score: {edge['confidence']}")

    return data

def extract_from_text(text: str, note_id: int) -> Dict:
    """
    Extract entities and relationships from text using local LLM.

    Args:
        text: Input text to analyze
        note_id: Associated note ID for provenance

    Returns:
        Extraction result with nodes and edges
    """
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=text)

    # Call LLM
    llm_output = call_local_llm(prompt)

    # Parse and validate
    extraction = parse_extraction_output(llm_output)

    # Add note_id to provenance
    for node in extraction["nodes"]:
        node["note_id"] = note_id

    return extraction

def update_graph_from_extraction(extraction: Dict, note_id: int, extraction_id: int):
    """
    Update NetworkX graph with extraction results.

    Args:
        extraction: Parsed extraction dict
        note_id: Source note ID
        extraction_id: Extract record ID
    """
    graph_store = get_graph_store()

    # Track created node IDs for edge creation
    node_label_to_id = {}

    # Add/update nodes
    for node_data in extraction["nodes"]:
        label = node_data["label"]
        node_id = generate_node_id(label)

        span_start, span_end = node_data["span"]
        provenance = [(note_id, span_start, span_end)]

        graph_store.add_node(
            node_id,
            label,
            node_data["type"],
            provenance=provenance,
            confidence=node_data["confidence"]
        )

        node_label_to_id[label] = node_id

    # Add edges
    for edge_data in extraction["edges"]:
        source_label = edge_data["source"]
        target_label = edge_data["target"]

        # Get node IDs (may need to generate if referenced node doesn't exist in this extraction)
        source_id = node_label_to_id.get(source_label, generate_node_id(source_label))
        target_id = node_label_to_id.get(target_label, generate_node_id(target_label))

        # Skip edge if either node doesn't exist in graph
        if not graph_store.graph.has_node(source_id) or not graph_store.graph.has_node(target_id):
            continue

        graph_store.add_edge(
            source_id,
            target_id,
            edge_data["type"],
            weight=edge_data["confidence"],
            extraction_id=extraction_id
        )

    # Save graph
    graph_store.save()

def process_note(note_id: int) -> Dict:
    """
    Full extraction pipeline for a note.

    Args:
        note_id: Note to process

    Returns:
        Processing result with stats
    """
    # Get note content
    note = get_note(note_id)
    if not note:
        raise ValueError(f"Note {note_id} not found")

    if note['processed']:
        return {"status": "already_processed", "note_id": note_id}

    content = note['content']

    # Extract
    try:
        extraction = extract_from_text(content, note_id)
    except Exception as e:
        return {
            "status": "extraction_failed",
            "note_id": note_id,
            "error": str(e)
        }

    # Store extract
    extraction_id = insert_extract(
        note_id,
        settings.llm_model,
        extraction,
        score=None  # Could compute average confidence
    )

    # Update graph
    try:
        update_graph_from_extraction(extraction, note_id, extraction_id)
    except Exception as e:
        return {
            "status": "graph_update_failed",
            "note_id": note_id,
            "extraction_id": extraction_id,
            "error": str(e)
        }

    # Mark as processed
    mark_note_processed(note_id)

    return {
        "status": "success",
        "note_id": note_id,
        "extraction_id": extraction_id,
        "nodes_extracted": len(extraction["nodes"]),
        "edges_extracted": len(extraction["edges"]),
        "summary": extraction.get("summary", "")
    }
