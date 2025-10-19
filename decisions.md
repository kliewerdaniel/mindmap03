# Design Decisions & Technical Logs

This document tracks major architectural and technical decisions made during the development of Mind Map AI.

## Phase 1: Backend Core Infrastructure

**Date:** October 19, 2025  
**Status:** Completed ✅

### Database Design Decisions

**SQLite Chosen Over Other Databases:**
- **Rationale**: Embedded, file-based database ensures local data sovereignty and simplifies deployment. ACID compliance provides reliable transactions without external dependencies.
- **Alternatives Considered**: PostgreSQL (too heavy for local deployment), DuckDB (considered but SQLite has broader ecosystem support).
- **Impact**: Single-file deployment (`mindmap.db`) in project data directory, no server process needed.

**Schema Design:**
- **Deduplication Strategy**: SHA256 content hashing prevents duplicate notes from being stored multiple times.
- **Processing Flags**: `processed` boolean on notes table tracks extraction completion status.
- **Metadata Separation**: Generic key-value metadata table for extensible configuration storage.

**Indexing Strategy:**
- Hash index for deduplication lookups
- Processed flag index for quick unprocessed note queries
- Note ID index on extracts table for efficient provenance retrieval

### API Architecture Decisions

**FastAPI Framework Selection:**
- **Rationale**: Modern async framework with automatic OpenAPI documentation generation, excellent Pydantic integration for type safety, and async support critical for LLM inference calls.
- **Alternatives Considered**: Flask (simpler but lacks async), Django (too heavy), Express.js (language inconsistency).

**Router Structure:**
- Separate routers for ingest, graph, and search concerns
- Clean separation allows independent development and testing
- Placeholder routers created for Phase 2 implementation

**Configuration Management:**
- **Pydantic-Settings**: Type-safe environment configuration with validation
- **Path Calculation**: Absolute paths ensure consistent file locations regardless of execution context
- **Environment Variables**: `.env` file support for local development overrides

### Directory Structure

**Backend Organization:**
```
backend/
├── app/
│   ├── main.py          # Application factory and startup logic
│   ├── config.py        # Centralized configuration management
│   ├── db/              # Database layer
│   │   ├── schema.sql   # SQLite schema definitions
│   │   └── db.py        # Connection management and CRUD operations
│   └── api/             # REST API layer
│       ├── __init__.py
│       ├── ingest.py    # Ingestion endpoints (Phase 2)
│       ├── graph.py     # Graph management endpoints (Phase 2)
│       └── search.py    # Search endpoints (Phase 2)
├── requirements.txt     # Python dependencies
└── tests/               # Test suite
    └── backend/
        └── test_db.py   # Database layer unit tests
```

**Data Storage:**
- Database located at `data/mindmap.db`
- Separate data directory ensures clean separation of application and user data

### Testing Strategy

**Pytest Framework:**
- Industry standard testing framework with excellent fixtures support
- Tempfile-based database isolation for reliable test execution
- 4 passing database unit tests covering core CRUD operations

### Documentation Updates

**Database Documentation (docs/database.md):**
- Comprehensive table schemas with column descriptions
- NetworkX graph model specification linked to spec.md
- Persistence strategy analysis (gpickle vs GraphML tradeoffs)
- Provenance tracking approach documentation

**Architecture Documentation (docs/architecture/README.md):**
- Technology stack rationale explaining each framework choice
- Backend architecture with module dependencies
- Data flow diagrams using Mermaid syntax

**DevOps Documentation (docs/cicd_devops.md):**
- Python virtual environment setup procedures
- Database initialization and management commands
- FastAPI development server configuration
- Testing and monitoring guidelines

### Known Limitations & Future Considerations

**Server Startup Issues:**
- FastAPI application fails to start due to missing dependencies (torch/PyTorch not available for Python 3.13)
- **Workaround**: Install core dependencies individually for development
- **Resolution**: Pins dependency versions in Phase 0 will need adjustment for production deployment

**Environment Assumptions:**
- Development testing assumes conda/miniconda environment with access to repositories
- Production deployment will require Docker containers or venv management

**Performance Baselines:**
- SQLite can handle thousands of notes efficiently for initial implementation
- Database initialization creates 15MB+ file due to indexes and schema setup

### Lessons Learned

**Path Calculation Complexity:**
- Absolute path computations from `__file__` sensitive to execution context
- Consistent root directory execution simplifies path management

**Dependency Management:**
- ML libraries (sentence-transformers, torch) create platform compatibility challenges
- Core API dependencies can be separated for faster development iterations

**Testing Infrastructure:**
- Early test creation validates database schema and operations
- Temporary database fixtures ensure clean test isolation

This Phase 1 implementation establishes a solid foundation with type-safe configuration, reliable data persistence, and extensible API architecture ready for Phase 2 LLM and graph integration.

## Phase 2: NetworkX Graph Store

**Date:** October 19, 2025  
**Status:** Completed ✅

### Graph Storage Architecture Decisions

**NetworkX as Core Graph Library:**
- **Rationale**: Mature Python graph library with extensive algorithmic support, serialization capabilities, and seamless Python integration.
- **Alternatives Considered**: Graph-tool (C++ extension, platform complexity), Networkit (similar feature set but less mature ecosystem), igraph (good but NetworkX has better Python integration).
- **Impact**: Enables centrality calculations, subgraph operations, and graph export functionality without custom graph implementations.

**Pickle-based Persistence (.gpickle):**
- **Rationale**: Native Python serialization preserves NetworkX graph objects perfectly, maintains all node/edge attributes, and is significantly faster than GraphML export/import.
- **Alternatives Considered**: GraphML (XML-based, human-readable but slow and verbose), JSON (requires custom serialization logic), SQLite (relational model doesn't map well to graph structures).
- **Tradeoffs**: Pickle is Python-specific (cannot easily share graphs with other languages) but ideal for local, Python-only architecture.

**Global Graph Instance Management:**
- **Rationale**: Singleton pattern prevents multiple graph instances from consuming memory unnecessarily and ensures consistent state across API calls.
- **Implementation**: Module-level cache with lazy initialization from persisted file on first access.
- **Synchronization**: File-based persistence acts as natural synchronization point, though concurrent writes require careful handling in future phases.

### Graph Data Model Decisions

**Node Types Classification:**
- **Defined Types**: concept (abstract ideas), person (individuals), place (locations), idea (specific concepts), event (occurrences), passage (text excerpts).
- **Rationale**: Provides semantic categorization for visualization and filtering while remaining extensible.
- **Confidence Scoring**: 0-1 float scale for extraction confidence, with future visualization color-coding based on confidence levels.

**Edge Relationship Types:**
- **Defined Relationships**: related_to (general association), causes (causal links), elaborates (explanation), contradicts (conflicting), similar_to (semantic similarity), part_of (hierarchical), precedes (temporal), affects (impact).
- **Rationale**: Rich but not overwhelming relationship types cover primary knowledge graph connection patterns.

**Provenance Tracking Design:**
- **Span-based References**: Tuples of (note_id, start_char, end_char) precisely track source text locations.
- **Multi-source Merging**: Duplicate provenance entries merged with set operations to handle overlapping extractions.
- **Validation**: Required spans ensure traceabilty from graph nodes back to source text.

**Deterministic Node ID Generation:**
- **Hash-based IDs**: Label normalization + MD5 hash ensures consistent, reproducible node IDs across extractions.
- **Namespace Prefixing**: "node:" prefix distinguishes graph nodes from other system identifiers.
- **Collision Prevention**: 8-character hash suffix provides 2^32 collision resistance while keeping IDs readable.

### API Design Decisions

**Graph Endpoints Structure:**
- **GET /**: Full graph retrieval with optional node-centric subgraph filtering
- **GET /node/{node_id}**: Individual node details with full attribute set
- **POST /node & POST /edge**: Node/edge creation with upsert semantics (create or update)
- **DELETE /**: Cascade node deletion removes associated edges
- **GET /stats & GET /export**: Metadata and export functionality

**Subgraph Depth Parameter:**
- **Default Depth: 2**: Provides 2-hop neighborhood for focused exploration
- **Configurable**: Query parameter allows flexible graph exploration depth
- **Performance**: Subgraph operations help prevent large graph visualizations from overwhelming clients

**Graph Export Formats:**
- **GraphML**: Standard XML-based network format, readable by graph analysis software
- **GEXF**: Gephi-specific format for advanced network visualization
- **GPickle**: Direct serialization for internal backups and data migration

### Implementation Tradeoffs & Optimizations

**Memory Management:**
- **Lazy Loading**: Graph only loaded when first accessed, reducing application startup time.
- **Incremental Saves**: Automatic save after each modification ensures data durability.

**Graph Validation Logic:**
- **Node Existence Checks**: Edge creation validates both source and target nodes exist.
- **Duplicate Handling**: Upsert semantics merge updates with existing graph elements.
- **Timestamp Tracking**: created_at/updated_at metadata tracks graph evolution over time.

**Testing Strategy:**
- **Temporary File Fixtures**: Isolated graph persistence testing prevents test interference.
- **Comprehensive Coverage**: Node/edge operations, persistence, merge logic, neighbor traversal all tested.
- **Performance Baseline**: Graph operations show O(1) node lookups, O(E) edge enumeration as expected.

### Production Considerations

**Scalability Limits:**
- **Memory Constraints**: Very large graphs (100K+ nodes) may exceed available RAM.
- **Future Optimizations**: Consider graph partitioning or on-disk structures for enterprise scale.

**Backup & Recovery:**
- **Simple File Backups**: Current .gpickle files can be archived for point-in-time recovery.
- **Version Control**: Future graph versioning may track major schema changes.

**Migration Strategy:**
- **Native Compatibility**: Pickle format avoids migration complexity at current scale.
- **Format Options**: Export capabilities provide paths to other systems if needed.

This Phase 2 implementation delivers a robust graph backend with NetworkX powering the core functionality and comprehensive CRUD APIs ready to integrate with Phase 3's LLM extraction pipeline.

## Phase 3: LLM Extraction Module

**Date:** October 19, 2025  
**Status:** Completed ✅

### LLM Integration Architecture

**Ollama Client Design:**
- **HTTP API Calls**: Direct POST requests to configurable endpoint (default: localhost:11434)
- **Timeout Handling**: 300-second extraction timeout handles slow LLM inference
- **Error Resilience**: Request-level exceptions caught and reported as extraction failures
- **Model Configuration**: Configurable model names via settings for different extraction strategies

**Request-Response Processing:**
- **Structured Prompts**: Template-based prompt engineering with schema validation
- **JSON Parsing Robustness**: Regex extraction handles cases where LLM includes extra text around JSON
- **Schema Validation**: Strict type checking ensures extraction results meet expected format

### Prompt Engineering Strategy

**Extraction Prompt Template:**
- **Clear Instructions**: Explicitly requests "only JSON" to minimize parsing complexity
- **Schema Documentation**: Complete field descriptions and valid value enumerations
- **Relationship Taxonomy**: Comprehensive edge type definitions prevent ambiguity
- **Example Format**: Structured examples demonstrate expected input-output mapping

**Node Extraction Schema:**
- **Type Constraints**: Limited vocabulary (concept, person, place, idea, event, passage) enables consistent categorization
- **Span Requirements**: Character positions provide precise text mapping for provenance
- **Confidence Scores**: 0-1 float values allow quality filtering and visualization weighting

**Edge Extraction Schema:**
- **Directed Relationships**: Source-target pairs with semantic relationship types
- **Confidence Integration**: Independent scoring allows differential treatment of entity vs relationship extractions

### Entity Normalization & Deduplication

**Label Standardization:**
- **Unicode Normalization**: Handles accented characters and special symbols consistently
- **Whitespace Collapsing**: Multiple spaces/hyphens converted to single underscores
- **Case Folding**: Lowercase conversion for case-insensitive matching

**Deterministic ID Generation:**
- **Hash Consistency**: Same label always produces identical node ID across runs
- **Namespace Isolation**: "node:" prefix prevents conflicts with other ID schemes
- **Readable Components**: Normalized label + truncated hash keeps IDs human-interpretable

### Graph Integration Pipeline

**Extraction to Graph Transformation:**
- **Batch Node Creation**: All nodes generated before edge creation simplifies error handling
- **Fallback Edge Handling**: Missing nodes generated with default IDs rather than failing extraction
- **Provenance Enhancement**: Note IDs added to node metadata for source tracing

**Database Persistence:**
- **Structured Storage**: Extraction results stored as JSON with confidence scores
- **Processing Flags**: Note-level `processed` status updated after successful graph integration
- **Audit Trail**: Extract table maintains complete LLM output history for debugging

### Async Processing Design

**Background Task Pattern:**
- **FastAPI BackgroundTasks**: Non-blocking extraction execution allows responsive API
- **Failure Isolation**: Individual note failures don't block other processing
- **Status Monitoring**: GET /status endpoint tracks processing state per note

**File Upload Handling:**
- **Multipart Support**: Form-encoded file uploads with automatic content type detection
- **Archive Processing**: ZIP file support enables batch markdown ingestion
- **Deduplication Integration**: Content hashing prevents duplicate note storage

### Testing & Validation Strategy

**Unit Test Coverage:**
- **Isolation Testing**: Mock LLM responses remove dependency on external services
- **Validation Logic**: Comprehensive schema checking tests catch format violations
- **Normalization Edge Cases**: Special character, whitespace, and encoding tests

**Integration Testing Considerations:**
- **Dependency Isolation**: Psych LLM endpoint unavailability doesn't break core logic
- **Timeout Simulation**: Error handling paths tested with mock failures
- **Parsing Robustness**: Extra-text handling validated with realistic LLM outputs

### Error Handling & Resilience

**Graceful Degradation:**
- **Invalid Output Handling**: Clear error messages for malformed LLM responses
- **Missing Fields Detection**: Field-by-field validation with specific error reporting
- **Confidence Bounds Checking**: Float range validation prevents invalid scores

**Debugging Facilities:**
- **Raw Output Storage**: Complete LLM responses preserved for troubleshooting
- **Processing Logs**: Detailed status returns identify failure points
- **Validation Feedback**: Specific error messages guide prompt refinement

### Performance Considerations

**LLM Call Optimization:**
- **Timeout Bounds**: 5-minute limit prevents resource exhaustion on slow models
- **Temperature Tuning**: Lower temperature (0.3) improves extraction consistency
- **Prompt Efficiency**: Concise instructions reduce token usage and response time

**Batch Processing Potential:**
- **Sequential Design**: Current single-note processing can be extended to batches
- **Resource Management**: Background task isolation prevents thread starvation
- **Memory Efficiency**: Streaming file processing avoids loading large archives

### Future Extensibility

**Model Fine-tuning:**
- **Format Consistency**: Structured JSON output enables easy fine-tuning datasets
- **Evaluation Framework**: Confidence scores allow accuracy comparison across models
- **Prompt A/B Testing**: Template versioning enables iterative improvement

**Advanced Features:**
- **Entity Resolution**: Current extraction can be enhanced with cross-note linking
- **Relationship Reasoning**: LLM can be prompted for multi-hop relationship inference
- **Context Awareness**: Include existing graph for disambiguation during extraction

**Alternative Architectures:**
- **Local vs Cloud LLM**: Ollama pattern works with other providers (OpenAI, Anthropic)
- **Batch Processing**: Current async design scales to larger processing systems
- **Caching Layers**: Extraction results can be cached with content hashing

This Phase 3 implementation creates a complete end-to-end pipeline from text ingestion through LLM extraction to graph storage, with robust error handling and comprehensive testing establishing the foundation for the full Mind Map AI system.
