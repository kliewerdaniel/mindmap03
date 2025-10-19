from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from .config import settings
from .db.db import init_database
from .api import ingest, graph, search

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute", "1000/hour"]  # Reasonable limits for API usage
)

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

# Rate limiting middleware
app.add_middleware(SlowAPIMiddleware)
app.state.limiter = limiter

# Rate limit exceeded handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return HTTPException(status_code=429, detail="Rate limit exceeded. Too many requests.")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_database()
    from .services.graph_store import init_graph
    from .services.embeddings import init_embeddings
    init_graph()
    init_embeddings()

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
