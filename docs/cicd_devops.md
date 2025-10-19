# CI/CD and DevOps Documentation

This document outlines the development operations, deployment, and maintenance procedures for Mind Map AI.

## Python Environment Setup

### Virtual Environment Management

**Creating and activating a virtual environment:**
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/MacOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

**Installing dependencies:**
```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (if requirements-dev.txt exists)
pip install -r requirements-dev.txt
```

**Verifying environment:**
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check virtual environment is active
which python  # Should point to .venv/bin/python
```

## Database Initialization

### SQLite Database Setup

**Automatic initialization:**
The database is automatically initialized when the FastAPI application starts. The schema is executed from `backend/app/db/schema.sql`.

**Manual initialization:**
```bash
cd backend
python -c "from app.db.db import init_database; init_database()"
```

**Database verification:**
```bash
# Check database file exists
ls -la ../data/mindmap.db

# Query database (requires sqlite3 CLI)
sqlite3 ../data/mindmap.db "SELECT name FROM sqlite_master WHERE type='table';"
```

**Backup database:**
```bash
cp ../data/mindmap.db ../data/mindmap.db.backup
```

## Running the Backend

### Development Server

**Start development server with auto-reload:**
```bash
cd backend
uvicorn app.main:app --reload
```

**Specify host and port:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**With logging:**
```bash
uvicorn app.main:app --reload --log-level info
```

### Production Server

**Using uvicorn directly:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Using gunicorn (with uvicorn workers):**
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## API Testing

### Health Check

**Test server is running:**
```bash
curl http://localhost:8000/health
# Expected response: {"status": "healthy"}
```

**API documentation:**
Open browser to: http://localhost:8000/docs

### Basic API Tests

**Test root endpoint:**
```bash
curl http://localhost:8000/
# Expected response: {"message": "Mind Map AI API", "version": "0.1.0"}
```

**Test CORS headers:**
```bash
curl -I http://localhost:8000/health
# Should include: Access-Control-Allow-Origin: *
```

## Testing

### Running Tests

**Run all tests:**
```bash
cd backend
pytest tests/ -v
```

**Run database tests specifically:**
```bash
pytest tests/backend/test_db.py -v
```

**Run tests with coverage:**
```bash
pytest tests/ --cov=app --cov-report=html
# View coverage report in htmlcov/index.html
```

**Run specific test function:**
```bash
pytest tests/backend/test_db.py::test_insert_note -v
```

### Test Configuration

**Environment variables for testing:**
```bash
export TESTING=true
export DATABASE_URL=sqlite:///./test.db
```

## Docker Support (Optional)

### Building Docker Image

**Build backend image:**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t mindmap-ai-backend .
```

### Running with Docker

```bash
# Run backend
docker run -p 8000:8000 mindmap-ai-backend

# With volume mounts for data persistence
docker run -p 8000:8000 -v $(pwd)/data:/app/data mindmap-ai-backend
```

### Docker Compose (if using)

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - ENVIRONMENT=docker

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
```

## Monitoring and Logging

### Application Logs

**Default logging:**
- INFO level for production
- DEBUG level for development (--log-level debug)

**Access logs:**
- Automatically included with uvicorn

### Health Monitoring

**Custom health endpoint:**
```python
# app/main.py
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Database health check:**
```python
@app.get("/health/db")
async def db_health_check():
    try:
        # Test database connection
        conn = get_connection()
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}
```

## Environment Management

### Configuration Files

**.env file structure:**
```bash
# LLM Configuration
LLM_ENDPOINT=http://localhost:11434/api/generate
LLM_MODEL=llama3
EMBEDDING_ENDPOINT=http://localhost:11434/api/embeddings
EMBEDDING_MODEL=all-minilm

# Database (SQLite defaults to file-based)
DATABASE_URL=sqlite:///./data/mindmap.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000

# Processing
MAX_BATCH_SIZE=10
EXTRACTION_TIMEOUT=300
```

**Loading environment variables:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_endpoint: str = "http://localhost:11434/api/generate"
    api_host: str = "0.0.0.0"
    # ... other settings

    class Config:
        env_file = ".env"
```

## Troubleshooting

### Common Issues

**Database connection errors:**
```bash
# Check file permissions
ls -la ../data/mindmap.db

# Recreate database
rm ../data/mindmap.db
python -c "from app.db.db import init_database; init_database()"
```

**LLM connection errors:**
```bash
# Test Ollama is running
curl http://localhost:11434/api/tags

# Check LLM endpoint configuration
curl http://localhost:11434/api/generate -d '{"model": "llama3", "prompt": "test"}'
```

**Port binding errors:**
```bash
# Check if port is in use
lsof -i :8000

# Kill process using port
kill -9 <PID>
```

**Dependency errors:**
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Check for conflicts
pip check
```

## Development Workflow

### Local Development
1. Create/activate virtual environment
2. Install dependencies
3. Run database initialization
4. Start development server
5. Run tests
6. Make code changes with auto-reload

### Feature Development
1. Create feature branch
2. Make changes
3. Write tests
4. Run tests and linting
5. Submit pull request

### Release Process
1. Update version numbers
2. Run full test suite
3. Build Docker images
4. Tag and release
5. Deploy to production

This CI/CD setup ensures consistent development and deployment processes while maintaining the local-first architecture of Mind Map AI.
