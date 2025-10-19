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
