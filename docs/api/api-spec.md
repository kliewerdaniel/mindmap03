# API Specification

## Search Endpoints

### POST /api/search/semantic

Semantic search across notes and/or nodes.

**Request Body:**
```json
{
  "q": "how does sleep affect productivity",
  "top_k": 10,
  "search_type": "both"
}
```

**Parameters:**
- `q`: Search query (required)
- `top_k`: Number of results (default: 10)
- `search_type`: Scope - "notes", "nodes", or "both" (default: "both")

**Response:**
```json
{
  "query": "how does sleep affect productivity",
  "results": [
    {
      "type": "node",
      "id": "node:sleep_quality_a3f9e2b1",
      "content": "sleep quality",
      "score": 0.92,
      "metadata": {
        "node_type": "concept",
        "provenance_count": 3
      }
    },
    {
      "type": "note",
      "id": "42",
      "content": "I've noticed that when I sleep poorly, my work performance drops significantly...",
      "score": 0.88,
      "metadata": {
        "filename": "journal-2024-01-15.md",
        "created_at": "2024-01-15T10:30:00"
      }
    }
  ],
  "total": 2
}
```

### GET /api/search/related/{node_id}

Find semantically related nodes.

**Parameters:**
- `node_id`: Source node ID
- `top_k`: Number of results (default: 5)

**Response:**
```json
{
  "source_node": "node:sleep_quality_a3f9e2b1",
  "related_nodes": [
    {
      "node_id": "node:rest_patterns_b2c4d5e6",
      "label": "rest patterns",
      "score": 0.89
    },
    {
      "node_id": "node:circadian_rhythm_c3d4e5f6",
      "label": "circadian rhythm",
      "score": 0.85
    }
  ]
}
