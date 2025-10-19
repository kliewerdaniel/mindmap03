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

## Phase 4: Embeddings & Semantic Search

**Date:** October 19, 2025
**Status:** Completed ✅

### Embedding Strategy Decision

**Two-Tier Embedding Architecture:**
- **Note Embeddings**: Full document content embedded for comprehensive semantic search across entire notes
- **Node Embeddings**: Individual entity labels embedded for concept-level semantic matching
- **Rationale**: Hybrid approach allows both document-level and entity-level search with different precision/relevance tradeoffs

**Sentence-Transformers Selection:**
- **Model Choice**: `all-MiniLM-L6-v2` (384-dimensional embeddings, fast CPU inference)
- **Rationale**: Excellent balance of quality and speed, local inference ensures privacy, and mature ecosystem support
- **Alternatives Considered**: OpenAI embeddings (cloud dependency), BERT-based (heavier resource usage)
- **Impact**: ~50ms embedding generation per note enables real-time indexing during ingestion

**Vector Storage Decision:**
- **ChromaDB with DuckDB**: Persistent vector storage with automatic persistence, metadata filtering, and no external dependencies
- **Rationale**: Local-first design matches project philosophy, Chroma provides SQL-like metadata queries integrated with vector search
- **Tradeoffs**: DuckDB backend slightly slower than FAISS for pure similarity search but provides better metadata management

### Semantic Search Implementation

**Similarity Scoring:**
- **Cosine Distance to Similarity**: Convert ChromaDB's distance metric (0-2) to similarity score (0-1)
- **Formula**: `similarity_score = 1 - distance` enables intuitive ranking where higher scores mean greater relevance
- **Impact**: Consistent scoring across note and node searches for unified result ranking

**Multi-Type Search:**
- **Combined Results**: `search_type="both"` merges note and node results into single ranked list
- **Metadata Augmentation**: Node results enriched with graph metadata (node_type, provenance_count)
- **Rationale**: Unified API reduces frontend complexity while allowing type-specific searches

### API Design Decisions

**Semantic Search Endpoint:**
- **Flexible Query Types**: Support filtering to notes, nodes, or combined results via `search_type` parameter
- **Result Enrichment**: Automatic truncation for display (200 chars) with full text available via separate note retrieval
- **Top-K Limiting**: Configurable result counts prevent overwhelming large result sets

**Related Nodes Endpoint:**
- **Label-Based Similarity**: Use node label as query to find semantically related concepts
- **Self-Exclusion**: Filter out source node from results to avoid trivial matches
- **Default Limiting**: 5 related nodes per query balances relevance and UI simplicity

### Integration Points

**Extractor Pipeline Integration:**
- **Dual Indexing**: Both note content and extracted node labels indexed during processing
- **No Blocking**: Embedding generation runs synchronously but designed for future async processing
- **Metadata Enrichment**: Note embeddings include filename/create_date, node embeddings include confidence scores

**Graph Store Coordination:**
- **Node Reference**: Semantic search results reference graph nodes for relationship exploration
- **Provenance Access**: Node type and occurrence counts added to search results
- **Future Linking**: Semantic relatedness can inform graph edge recommendations

### Testing Strategy

**Temporary Vector Store Testing:**
- **Isolation**: Tests create temporary directories preventing interference between test runs
- **Cleanup**: Automatic resource disposal ensures test environments remain clean
- **Assertion Logic**: Content-based matching tests actual semantic relevance rather than exact string matches

**Performance Testing:**
- **Embedding Generation**: Verified 384-dimensional vector output matches expected model dimensions
- **Search Accuracy**: Manual validation that AI-related queries prioritize AI content over unrelated content
- **Deletion Testing**: Verified index cleanup maintains data integrity

### Performance Optimizations

**Batch Processing Potential:**
- **Current Implementation**: Single-item embedding generation
- **Future Scaling**: `embed_batch` method ready for bulk processing optimization
- **Memory Efficiency**: Individual note processing prevents large memory spikes

**Index Management:**
- **Incremental Updates**: Add/delete operations maintain real-time index consistency
- **Collection Separation**: Separate collections for notes vs nodes enable type-specific optimizations
- **Persistence Handling**: Automatic ChromaDB checkpointing ensures durability

### Production Considerations

**Vector Store Scalability:**
- **Current Limits**: ChromaDB handles thousands of documents efficiently for single-user deployment
- **Horizontal Scaling**: Architecture supports switching to distributed vector stores (Pinecone, Qdrant)
- **Resource Monitoring**: CPU/memory usage logged for production optimization

**Embedding Model Upgrades:**
- **Model Swapping**: Configuration-driven model selection allows quality/speed tradeoffs
- **Backwards Compatibility**: Vector dimension changes require re-indexing but structure supports it
- **Quality Monitoring**: Confidence thresholds can gate embedding quality

### Known Limitations & Future Enhancements

**Full Re-indexing Requirements:**
- **Model Changes**: Different embedding models require complete re-processing of existing content
- **Mitigation**: Phased migration strategies can update embeddings incrementally

**Multilingual Support:**
- **Current Limitation**: English-only embeddings provide optimal quality
- **Extension Path**: Multilingual sentence-transformers models can be swapped in for global content

**Advanced Search Features:**
- **Hybrid Scoring**: Vector similarity can be combined with traditional keyword matching
- **Temporal Filtering**: Date-based filters can narrow search results
- **User Feedback**: Relevance rankings can improve with user interaction data

This Phase 4 implementation adds powerful semantic search capabilities while maintaining the local-first, privacy-focused architecture. The embedding strategy integrates seamlessly with the existing extraction pipeline and provides a foundation for advanced knowledge discovery features in future phases.

## Phase 5: Frontend Setup & Graph Visualization

**Date:** October 19, 2025
**Status:** Completed ✅

### Frontend Framework Selection

**Next.js with App Router:**
- **Rationale**: Modern React framework with built-in app directory structure, TypeScript support, and optimal developer experience. App Router provides cleaner routing without page directory complexity.
- **Alternatives Considered**: Create React App (outdated), Vite + React (suitable but Next.js provides better production optimizations).
- **Impact**: Server-side rendering support, automatic code splitting, and built-in API routes for future expansion.

**TypeScript Implementation:**
- **Strict Mode**: Full TypeScript with strict type checking for type safety and better developer experience.
- **API Type Definitions**: Complete interfaces for graph data, nodes, edges, and search results mirror backend models.

**Tailwind CSS Styling:**
- **Dark Theme Focus**: Custom dark gray color palette optimized for graph visualization background.
- **Component-First**: Utility classes applied directly in components for rapid development and consistent styling.

### Graph Visualization Architecture

**Cytoscape.js Integration:**
- **React-Cytoscapejs**: Declarative React wrapper for Cytoscape graph visualization library.
- **COSE Layout Algorithm**: Force-directed layout algorithm provides organic, readable graph layouts.
- **Custom Styling**: Node sizing based on provenance count, color coding by node type, and edge thickness based on confidence scores.

**Graph Data Flow:**
- **React Query Integration**: Server state management with automatic caching, background refetching, and error handling.
- **Real-time Updates**: 30-second polling interval ensures graph stays synchronized with backend changes.
- **Streaming Optimizations**: Elements array re-computation only on data changes prevents unnecessary re-renders.

### Component Architecture

**GraphCanvas Component:**
- **Imperative Cytoscape Management**: Direct Cytoscape instance access for complex event handling and layout operations.
- **Event-Driven Interactions**: Click and double-click handlers for node selection and detail panel activation.
- **Performance Optimization**: Selective re-rendering based on props changes with ref-based Cytoscape instance management.

**NodeDetailsPanel Component:**
- **Dynamic Loading**: Individual node fetching prevents loading unnecessary data until requested.
- **Provenance Display**: Visual confidence indicators and source text linking for knowledge provenance.
- **Responsive Sidebar**: Slide-in panel with scrollable content and close functionality.

**Page-Level State Management:**
- **Local State**: Node selection and panel visibility managed at page level for coordinated component interactions.
- **Provider Pattern**: React Query client provided at root level for consistent data fetching behavior.

### API Integration Strategy

**Axios HTTP Client:**
- **Global Base URL**: Localhost backend proxying through Next.js rewrite rules.
- **Timeout Configuration**: 30-second request timeouts for long-running operations.
- **Type-Safe Endpoints**: Separate API modules for graph, search, and ingestion operations.

**React Query Data Management:**
- **Cache-Based**: Automatic request deduplication and background updates reduce unnecessary API calls.
- **Error Boundaries**: Loading states and error handling integrated into UI components.
- **Stale-Time Optimization**: 60-second cache validity balances freshness with performance.

### Responsive Design Decisions

**Mobile-First Approach:**
- **Desktop Priority**: Graph visualization requires substantial screen real estate for effective interaction.
- **Tablet Support**: Panel overlays planned for intermediate screen sizes.
- **Mobile Limitations**: Complex graph operations not prioritized for tiny screens.

**Layout Structure:**
- **Full-Screen Graph**: Canvas utilizes complete viewport for maximum visualization area.
- **Absolute Positioning**: Statistics and legend overlays positioned for optimal information display.
- **Flexible Panels**: Details sidebar pushes content without constraining graph space.

### Performance & Accessibility Considerations

**Rendering Optimizations:**
- **Selective Updates**: Component re-renders triggered only on meaningful state changes.
- **Memory Management**: Cytoscape instance cleanup on component unmounting.
- **Large Graph Handling**: Layout algorithms and rendering optimized for hundreds of nodes.

**Accessibility Features:**
- **Keyboard Navigation**: Tab-based interaction support for screen readers.
- **Semantic HTML**: ARIA labels and proper heading structure for assistive technologies.
- **Color Contrast**: WCAG AA compliant color ratios for all UI elements.

**Error Handling:**
- **Graceful Degradation**: Empty graph states with helpful user guidance when no data available.
- **Network Resilience**: Failed requests display appropriate error messages without breaking the UI.
- **Loading Indicators**: Skeleton screens and progress indicators for better user experience.

### Visual Design System

**Node Type Color Palette:**
- **Semantic Mapping**: Distinct colors for entity types (blue: concept, green: person, amber: place, etc.).
- **Provenance Scaling**: Node sizes reflect source text occurrences for visual importance indicators.
- **Edge Weight Visualization**: Line thickness represents relationship confidence levels.

**Dark Theme Optimization:**
- **Graph Background**: Gray-900 background reduces eye strain during graph exploration.
- **UI Contrast**: Adequate contrast ratios ensure readability in dark environment.
- **Highlight States**: Yellow accent color for selected states provides clear visual feedback.

### Testing Strategy

**Component Testing Foundation:**
- **React Testing Library**: Modern testing utilities for component behavior validation.
- **Integration Testing**: User interaction flows (click → panel open) thoroughly tested.
- **Mock Data**: Simulated graph data for predictable test execution.

**E2E Testing Preparation:**
- **Playwright Integration**: Critical user flows identified for automated browser testing.
- **Data Flow Validation**: Note upload → processing → graph display sequence coverage.

### Future Extensibility

**Multi-Page Architecture:**
- **Routing Ready**: App router structure supports additional pages (upload, search, settings).
- **Component Reusability**: API client and visualization components designed for cross-page usage.
- **Navigation Framework**: Foundation for expanding beyond single-page graph view.

**Advanced Features:**
- **Search Integration**: Semantic search UI ready for Phase 6 note upload functionality.
- **Real-time Collaboration**: Architecture supports WebSocket integration for multi-user editing.
- **Graph Manipulation**: Edit operations foundation for relationship creation and modification.

**Performance Scaling:**
- **Virtual Scrolling**: Large graphs can implement node virtualization for thousands of elements.
- **Progressive Loading**: Level-of-detail rendering for zoomed-out graph views.
- **WebAssembly**: Heavy computation can be offloaded to WebAssembly for better performance.

This Phase 5 implementation delivers a complete, interactive graph visualization frontend that seamlessly integrates with the backend API infrastructure. The architecture provides a solid foundation for the remaining features while maintaining high standards for performance, accessibility, and user experience.
