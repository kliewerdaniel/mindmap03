# API Documentation

This directory contains comprehensive API documentation for the Mind Map AI system.

## Overview

The Mind Map AI API provides RESTful endpoints for:
- Knowledge graph management
- Entity and relationship extraction
- Semantic search functionality
- Graph visualization and export
- Note upload and processing

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API operates without authentication for local development. Production deployment should include proper authentication mechanisms.

## Core Endpoints

### Graph Management
- `GET /graphs` - List all knowledge graphs
- `POST /graphs` - Create a new knowledge graph
- `GET /graphs/{graph_id}` - Retrieve specific graph
- `PUT /graphs/{graph_id}` - Update graph metadata
- `DELETE /graphs/{graph_id}` - Delete graph

### Entity Operations
- `GET /graphs/{graph_id}/entities` - List entities in graph
- `POST /graphs/{graph_id}/entities` - Add entity to graph
- `GET /graphs/{graph_id}/entities/{entity_id}` - Get entity details
- `PUT /graphs/{graph_id}/entities/{entity_id}` - Update entity
- `DELETE /graphs/{graph_id}/entities/{entity_id}` - Remove entity

### Relationship Operations
- `GET /graphs/{graph_id}/relationships` - List relationships
- `POST /graphs/{graph_id}/relationships` - Create relationship
- `DELETE /graphs/{graph_id}/relationships/{rel_id}` - Remove relationship

### Search & Query
- `POST /search/semantic` - Semantic similarity search
- `POST /search/entities` - Entity-based search
- `GET /search/suggestions` - Search suggestions

### Note Processing
- `POST /notes/upload` - Upload and process notes
- `GET /notes/{note_id}` - Retrieve processed note
- `POST /notes/{note_id}/extract` - Manual extraction trigger

## Response Formats

All API responses follow a consistent format:

```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

Error responses:
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {}
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Data Models

### Entity
```json
{
  "id": "string",
  "label": "string",
  "type": "PERSON|ORGANIZATION|CONCEPT|LOCATION|EVENT",
  "properties": {
    "description": "string",
    "confidence": 0.95,
    "source": "note_id"
  },
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

### Relationship
```json
{
  "id": "string",
  "source_id": "string",
  "target_id": "string",
  "type": "RELATED_TO|WORKS_FOR|LOCATED_IN|PART_OF",
  "properties": {
    "description": "string",
    "confidence": 0.87,
    "source": "note_id"
  },
  "created_at": "2024-01-01T12:00:00Z"
}
```

### Graph
```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "entity_count": 150,
  "relationship_count": 230,
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:
- 100 requests per minute for read operations
- 10 requests per minute for write operations
- 5 requests per minute for note processing

## File Upload

Note upload supports multiple file formats:
- Plain text (.txt)
- Markdown (.md)
- PDF (.pdf) - with text extraction
- Word documents (.docx)
- HTML files (.html)

Maximum file size: 10MB per upload.

## WebSocket Support

For real-time graph updates:
```
ws://localhost:8000/ws/graph/{graph_id}
```

## SDKs and Libraries

- Python SDK (planned)
- JavaScript/TypeScript SDK (planned)
- REST API (current)

## Examples

See the [API Examples](./examples/) directory for comprehensive usage examples in multiple programming languages.

## Support

For API support and questions:
- Check the [troubleshooting guide](./troubleshooting.md)
- Review [common issues](./issues.md)
- Open an issue on GitHub
