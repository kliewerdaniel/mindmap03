from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .db.db import init_database
from .api import ingest, graph, search

app = FastAPI(
    title="Mind Map AI",
    description="Local LLM-powered personal knowledge graph",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_database()
    # Initialize graph store (will be implemented in Phase 2)
    # from .services.graph_store import init_graph
    # init_graph()

# Include routers
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingestion"])
app.include_router(graph.router, prefix="/api/graph", tags=["graph"])
app.include_router(search.router, prefix="/api/search", tags=["search"])

@app.get("/")
async def root():
    return {"message": "Mind Map AI API", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
