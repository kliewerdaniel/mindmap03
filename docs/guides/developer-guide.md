# Developer Guide - Mind Map AI

This guide provides comprehensive information for developers working on the Mind Map AI codebase.

## Development Environment Setup

### Prerequisites

**Required Software:**
- Python 3.8+
- Node.js 18+
- Git
- Ollama or Llama.cpp (for LLM functionality)

**Recommended Tools:**
- Visual Studio Code
- Docker (for containerized development)
- SQLite Browser (for database inspection)

### Initial Setup

1. **Clone and navigate:**
```bash
git clone <repository-url>
cd mindmap-ai
```

2. **Backend setup:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

3. **Install additional development dependencies:**
```bash
pip install -r requirements-dev.txt
```

4. **Frontend setup:**
```bash
cd ../frontend
npm install
```

5. **Environment configuration:**
```bash
# Copy and modify environment files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

## Project Structure

```
mindmap-ai/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── api/            # API route handlers
│   │   │   └── v1/         # API version 1 endpoints
│   │   ├── core/           # Business logic and services
│   │   │   ├── config.py   # Application configuration
│   │   │   ├── llm/        # LLM integration modules
│   │   │   ├── graph/      # NetworkX graph operations
│   │   │   └── search/     # Vector search functionality
│   │   ├── db/             # Database models and connections
│   │   ├── models/         # Pydantic models
│   │   └── services/       # Service layer
│   ├── tests/              # Test suite
│   └── requirements*.txt    # Python dependencies
├── frontend/               # Next.js application
│   ├── components/         # React components
│   ├── pages/              # Next.js pages
│   ├── public/             # Static assets
│   ├── styles/             # CSS and styling
│   └── package.json        # Node.js dependencies
├── docs/                   # Documentation
├── data/                   # Data storage
└── scripts/                # Utility scripts
```

## Backend Development

### FastAPI Application Structure

The backend follows a modular architecture:

- **API Layer**: Request/response handling and validation
- **Service Layer**: Business logic and orchestration
- **Data Layer**: Database and external service interactions
- **Core Layer**: Shared utilities and configurations

### Key Development Patterns

#### Dependency Injection
```python
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    # Database
    db = providers.Singleton(Database, db_url=config.database_url)

    # Services
    graph_service = providers.Singleton(
        GraphService,
        db=db,
        config=config
    )
```

#### Error Handling
```python
from fastapi import HTTPException

class GraphService:
    async def get_graph(self, graph_id: str) -> Graph:
        try:
            graph = await self.repository.get(graph_id)
            if not graph:
                raise HTTPException(
                    status_code=404,
                    detail=f"Graph {graph_id} not found"
                )
            return graph
        except DatabaseError as e:
            raise HTTPException(
                status_code=500,
                detail="Internal database error"
            )
```

#### Database Models
```python
from sqlalchemy import Column, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Entity(Base):
    __tablename__ = "entities"

    id = Column(String, primary_key=True)
    label = Column(String, nullable=False)
    type = Column(String, nullable=False)  # PERSON, ORG, etc.
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Testing

#### Running Tests
```bash
# Backend tests
cd backend
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html

# Frontend tests
cd ../frontend
npm test
```

#### Writing Tests
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_graph(client: AsyncClient):
    response = await client.post(
        "/api/v1/graphs",
        json={"name": "Test Graph", "description": "A test graph"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Graph"
```

## Frontend Development

### Next.js Architecture

The frontend uses App Router architecture with:
- **Server Components**: For initial page loads and SEO
- **Client Components**: For interactive features
- **API Routes**: For frontend-specific endpoints

### Component Structure

```typescript
// components/GraphVisualization.tsx
'use client'

import { useCytoscape } from '@/hooks/useCytoscape'
import { GraphData } from '@/types/graph'

interface GraphVisualizationProps {
  data: GraphData
  onNodeSelect?: (node: Node) => void
}

export function GraphVisualization({
  data,
  onNodeSelect
}: GraphVisualizationProps) {
  const cy = useCytoscape(data, { onNodeSelect })

  return <div id="cy" className="w-full h-full" />
}
```

### State Management

```typescript
// lib/store/graphStore.ts
import { create } from 'zustand'

interface GraphState {
  currentGraph: Graph | null
  loading: boolean
  error: string | null
  setCurrentGraph: (graph: Graph) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export const useGraphStore = create<GraphState>((set) => ({
  currentGraph: null,
  loading: false,
  error: null,
  setCurrentGraph: (graph) => set({ currentGraph: graph }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error })
}))
```

## Data Layer Development

### Graph Operations

```python
# app/core/graph/operations.py
from networkx import DiGraph, write_graphml, read_graphml
from typing import List, Optional

class GraphOperations:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.graph = DiGraph()

    def add_entity(self, entity: Entity) -> None:
        """Add an entity node to the graph."""
        self.graph.add_node(
            entity.id,
            label=entity.label,
            type=entity.type,
            confidence=entity.confidence
        )

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        confidence: float
    ) -> None:
        """Add a relationship edge to the graph."""
        self.graph.add_edge(
            source_id,
            target_id,
            type=rel_type,
            confidence=confidence
        )

    def save_graph(self, filename: str) -> None:
        """Persist graph to disk."""
        path = f"{self.storage_path}/{filename}.graphml"
        write_graphml(self.graph, path)

    def load_graph(self, filename: str) -> None:
        """Load graph from disk."""
        path = f"{self.storage_path}/{filename}.graphml"
        self.graph = read_graphml(path)
```

### Vector Operations

```python
# app/core/search/vectors.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def build_index(self, texts: List[str]) -> None:
        """Build FAISS index from text documents."""
        embeddings = self.model.encode(texts)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        self.metadata = texts

    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for similar texts."""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if idx != -1:  # Valid index
                results.append(self.metadata[idx])

        return results
```

## API Development

### Endpoint Structure

```python
# app/api/v1/endpoints/graphs.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

router = APIRouter()

@router.get("/graphs", response_model=List[GraphResponse])
async def list_graphs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all knowledge graphs."""
    graphs = await graph_service.get_graphs(db, skip=skip, limit=limit)
    return graphs

@router.post("/graphs", response_model=GraphResponse, status_code=201)
async def create_graph(
    graph: GraphCreate,
    db: Session = Depends(get_db)
):
    """Create a new knowledge graph."""
    return await graph_service.create_graph(db, graph)

@router.get("/graphs/{graph_id}", response_model=GraphResponse)
async def get_graph(
    graph_id: str,
    db: Session = Depends(get_db)
):
    """Retrieve a specific graph."""
    graph = await graph_service.get_graph(db, graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    return graph
```

## Configuration Management

### Backend Configuration

```python
# app/core/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Settings
    api_host: str = "localhost"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Database
    database_url: str = "sqlite:///./data/metadata.db"

    # LLM Settings
    llm_provider: str = "ollama"  # ollama, llama_cpp
    llm_model: str = "llama2"
    llm_base_url: str = "http://localhost:11434"

    # Storage
    graph_storage_path: str = "./data/graphs"
    embedding_storage_path: str = "./data/embeddings"

    # Processing
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    supported_formats: List[str] = [".txt", ".md", ".pdf", ".docx"]

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

## Logging and Monitoring

### Structured Logging

```python
import logging
import json
from pythonjsonlogger import jsonlogger

# Configure JSON logging
logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Processing note", extra={
    "note_id": "123",
    "file_size": 1024,
    "processing_time": 1.5
})
```

## Performance Optimization

### Caching Strategy

```python
from functools import lru_cache
import asyncio
from typing import Dict, Any

class CacheManager:
    def __init__(self):
        self.llm_cache: Dict[str, Any] = {}
        self.graph_cache: Dict[str, DiGraph] = {}

    @lru_cache(maxsize=1000)
    def get_llm_response(self, prompt: str) -> str:
        """Cache LLM responses to avoid redundant calls."""
        return self._call_llm(prompt)

    async def get_graph_cached(self, graph_id: str) -> Optional[DiGraph]:
        """Cache loaded graphs for faster access."""
        if graph_id in self.graph_cache:
            return self.graph_cache[graph_id]

        graph = await self._load_graph(graph_id)
        if graph:
            self.graph_cache[graph_id] = graph
        return graph
```

## Deployment

### Development Deployment

```bash
# Start backend with auto-reload
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend
cd frontend
npm run dev
```

### Production Deployment

```dockerfile
# Dockerfile.backend
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

## Contributing Guidelines

### Code Style

**Backend (Python):**
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Maximum line length: 88 characters
- Use black for code formatting

**Frontend (TypeScript):**
- Follow ESLint configuration
- Use Prettier for code formatting
- Interface naming: PascalCase
- Component naming: PascalCase
- Hook naming: camelCase

### Git Workflow

1. **Create feature branch:** `git checkout -b feature/amazing-feature`
2. **Make changes and commit:** `git commit -m "Add amazing feature"`
3. **Push branch:** `git push origin feature/amazing-feature`
4. **Create Pull Request**

### Testing Requirements

- **Unit tests**: Minimum 80% code coverage
- **Integration tests**: Test API endpoints and database operations
- **E2E tests**: Test complete user workflows
- **Performance tests**: Ensure acceptable response times

## Debugging

### Common Debugging Techniques

**Backend Debugging:**
```python
import pdb; pdb.set_trace()  # Python debugger
import ipdb; ipdb.set_trace()  # IPython debugger
```

**Frontend Debugging:**
```javascript
console.log('Debug info:', data)
debugger; // Breakpoint in browser
```

**Database Debugging:**
```sql
-- Enable query logging
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

## Performance Profiling

### Backend Profiling

```python
import cProfile
import pstats

# Profile code execution
profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = some_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Frontend Profiling

Use React DevTools Profiler or browser performance tools to identify slow components and optimize rendering.

## Security Considerations

### Input Validation

```python
from pydantic import BaseModel, validator
from fastapi import HTTPException

class NoteUpload(BaseModel):
    filename: str
    content: str

    @validator('filename')
    def validate_filename(cls, v):
        if not v.endswith(('.txt', '.md', '.pdf')):
            raise ValueError('Unsupported file type')
        return v

    @validator('content')
    def validate_content(cls, v):
        if len(v) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError('File too large')
        return v
```

### SQL Injection Prevention

```python
# Use parameterized queries
query = "SELECT * FROM entities WHERE type = :entity_type"
result = db.execute(query, {"entity_type": "PERSON"})

# Avoid string formatting for queries
# BAD: f"SELECT * FROM entities WHERE type = '{user_input}'"
```

This developer guide provides the foundation for contributing to Mind Map AI. For specific implementation details, refer to the inline code documentation and existing patterns in the codebase.
