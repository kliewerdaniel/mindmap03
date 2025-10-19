# Database Design

## SQLite Schema

### Tables

#### notes
Stores raw markdown/text content with metadata.

| Column      | Type     | Description                                      |
|-------------|----------|--------------------------------------------------|
| id          | INTEGER  | Primary key, autoincrement                       |
| filename    | TEXT     | Not null - Original filename                      |
| content     | TEXT     | Not null - Raw markdown/text content             |
| created_at  | TIMESTAMP| Auto-generated creation timestamp                |
| updated_at  | TIMESTAMP| Auto-updated modification timestamp              |
| source_path | TEXT     | Optional - Original filesystem path              |
| hash        | TEXT     | Unique - SHA256 hash of content for deduplication |
| processed   | BOOLEAN  | Default 0 - Flag indicating extraction completion |

#### extracts
Stores LLM extraction results with provenance.

| Column          | Type      | Description                                      |
| ----------------|-----------|--------------------------------------------------|
| id              | INTEGER   | Primary key, autoincrement                       |
| note_id         | INTEGER   | Foreign key to notes.id                          |
| extractor_model | TEXT      | Not null - Model identifier (e.g., "llama3-8b") |
| extract_json    | TEXT      | Not null - Raw JSON output from LLM              |
| score           | REAL      | Optional - Confidence/quality score              |
| created_at      | TIMESTAMP | Auto-generated timestamp                         |

#### metadata
Key-value store for system metadata.

| Column     | Type      | Description                        |
|------------|-----------|------------------------------------|
| key        | TEXT      | Primary key - Metadata key         |
| value      | TEXT      | Metadata value                     |
| updated_at | TIMESTAMP | Auto-updated timestamp             |

### Indexes
- `idx_notes_hash` on notes(hash)
- `idx_notes_processed` on notes(processed)
- `idx_extracts_note_id` on extracts(note_id)

## NetworkX Graph Model

The knowledge graph is represented using NetworkX, a Python library for complex networks. Nodes represent entities and concepts, while edges represent relationships between them.

### Node Attributes
- `id`: Unique string identifier (e.g., "node:UUID" or "entity:<normalized_text>")
- `label`: Display name of the entity/concept
- `type`: Entity type (concept, person, place, idea, event, passage)
- `provenance`: List of (note_id, span_start, span_end) tuples indicating source locations
- `embedding`: Optional vector representation; stored separately in vector database
- `created_at`: Node creation timestamp
- `updated_at`: Node last modification timestamp

### Edge Attributes
- `type`: Relationship type (related_to, causes, elaborates, contradicts, similar_to, part_of)
- `weight`: Confidence score (float between 0-1)
- `extraction_id`: Reference to extracts table entry
- `provenance`: Source text spans supporting this relationship

### Graph Persistence Strategy

Two main serialization formats are used:

#### GPickle (.gpickle files)
- **Pros**: Fast loading/saving, preserves all Python object types, compact binary format
- **Cons**: Not human-readable, tied to Python pickle format
- **Use case**: Default persistence for rapid development and frequent saves

#### GraphML (.graphml files)
- **Pros**: Human-readable XML format, interoperable with other graph tools, standard format
- **Cons**: Larger file size, slower to serialize/deserialize
- **Use case**: Export/import for backup, visualization tools, or data exchange

Tradeoff: GPickle preferred for performance in production use, GraphML for human inspection and interoperability.

## Provenance Tracking

All knowledge graph elements (nodes and edges) maintain provenance information linking back to source documents and LLM extractions:

1. **Source Document Tracking**: Each note stored in SQLite with unique hash for deduplication
2. **Extraction Provenance**: Raw LLM outputs stored in extracts table with scores and timestamps
3. **Span References**: Character-level spans in source text stored with each extracted element
4. **Bidirectional Links**: Graph elements reference both extraction records and source document spans
5. **Timestamp History**: Creation and modification times tracked for audit and conflict resolution

This multi-layered provenance ensures full traceability from graph elements back to original source text, supporting:
- Content validation and confidence scoring
- Extract conflict resolution
- Source citation in user interfaces
- Audit trails for graph modifications
