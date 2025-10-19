from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # LLM Configuration
    llm_endpoint: str = "http://localhost:11434/api/generate"  # Default Ollama endpoint
    llm_model: str = "granite4:micro-h"
    embedding_endpoint: str = "http://localhost:11434/api/embeddings"
    embedding_model: str = "all-minilm"

    # Database Paths
    db_path: Path = Path(__file__).parent.parent.parent / "data" / "mindmap.db"
    graph_path: Path = Path(__file__).parent.parent.parent / "data" / "graph.gpickle"
    vector_db_path: Path = Path(__file__).parent.parent.parent / "data" / "vectors"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list = ["http://localhost:3000"]

    # Processing Configuration
    max_batch_size: int = 10
    extraction_timeout: int = 300  # seconds

    # Security Configuration
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB limit
    allowed_extensions: list = [".md", ".txt", ".zip"]
    disable_external_llm: bool = False  # Set to True to enforce local-only LLM usage

    class Config:
        env_file = ".env"

settings = Settings()
