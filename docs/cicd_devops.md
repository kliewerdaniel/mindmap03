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

## Docker Containerization

### Overview

Mind Map AI provides complete containerization with Docker and Docker Compose for both backend and frontend services, including Ollama for local LLM processing. The setup includes security hardening, health checks, and proper networking.

### Container Architecture

- **Backend**: Python FastAPI application with security middleware
- **Frontend**: Next.js React application with TypeScript
- **Ollama**: Local LLM server for embeddings and generation
- **Networking**: Isolated bridge network for inter-service communication
- **Volumes**: Persistent storage for data and Ollama models

### Prerequisites

**Install Docker and Docker Compose:**
```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Quick Start with Docker Compose

**Clone and navigate to project:**
```bash
git clone <repository-url>
cd mindmap03
```

**Start all services:**
```bash
docker-compose up -d
```

**View service status:**
```bash
docker-compose ps
# Expected output shows all services healthy
```

**View logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
```

**Stop services:**
```bash
docker-compose down
```

### Development with Docker Compose

**Build and start services:**
```bash
# Build with no cache
docker-compose build --no-cache

# Start services
docker-compose up -d

# Follow logs with timestamps
docker-compose logs -f -t
```

**Access services:**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Ollama**: http://localhost:11434

### Ollama Setup in Docker

**Pull required models:**
```bash
# Connect to Ollama container
docker-compose exec ollama bash

# Pull models
ollama pull granite4:micro-h  # For generation
ollama pull all-minilm        # For embeddings
```

**Verify models are loaded:**
```bash
ollama list
# Expected output shows installed models
```

### Environment Configuration

**Create environment file (.env):**
```bash
# Project root .env file
DISABLE_EXTERNAL_LLM=true
MAX_UPLOAD_SIZE=10485760
ALLOWED_EXTENSIONS=".md,.txt,.zip"
```

**Security settings:**
- `DISABLE_EXTERNAL_LLM`: Enforce local-only LLM processing
- `MAX_UPLOAD_SIZE`: Maximum file upload size in bytes
- `ALLOWED_EXTENSIONS`: Whitelist of allowed file extensions

### Production Deployment

**Production configuration:**
```bash
# Create production .env
cp .env.example .env.production

# Build production images
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**Production docker-compose.prod.yml:**
```yaml
version: '3.8'

services:
  backend:
    environment:
      - DISABLE_EXTERNAL_LLM=true
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  frontend:
    environment:
      - NEXT_PUBLIC_API_URL=https://yourdomain.com/api
```

### Manual Docker Commands

**Build individual images:**
```bash
# Backend
docker build -t mindmap-ai-backend ./backend

# Frontend
docker build -t mindmap-ai-frontend ./frontend
```

**Run individual containers:**
```bash
# Backend only
docker run -d -p 8000:8000 -v $(pwd)/data:/app/data mindmap-ai-backend

# Frontend only
docker run -d -p 3000:3000 mindmap-ai-frontend
```

**Ollama standalone:**
```bash
docker run -d -p 11434:11434 --name ollama ollama/ollama
```

### Container Management

**Health checks:**
```bash
# Check all service health
docker-compose ps

# Test backend health
curl http://localhost:8000/health

# Test frontend health
curl -I http://localhost:3000
```

**Container cleanup:**
```bash
# Stop and remove containers
docker-compose down

# Remove volumes
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Complete cleanup
docker system prune -af
```

### Troubleshooting Docker Issues

**Container startup failures:**
```bash
# Check logs
docker-compose logs backend

# Check container status
docker-compose ps

# Restart failing service
docker-compose restart backend
```

**Permission issues:**
```bash
# Fix data directory permissions
sudo chown -R 1000:1000 ./data

# Check Docker daemon status
sudo systemctl status docker
```

**Port conflicts:**
```bash
# Check port usage
lsof -i :8000

# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Change external port
```

**Memory/CPU issues:**
```bash
# Monitor resource usage
docker stats

# Adjust limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '2.0'
```

### Security Considerations

**Container security:**
- Non-root user execution in all containers
- Minimal base images (slim/alpine)
- No privileged containers
- Proper .dockerignore usage

**Network security:**
- Isolated network for inter-service communication
- No external exposure of sensitive services
- Proper CORS configuration

### Performance Optimization

**Build optimization:**
- Multi-stage builds for smaller images
- Proper layer caching with .dockerignore
- Dependency installation before code copy

**Runtime optimization:**
- Health checks for service orchestration
- Proper restart policies
- Resource limits and reservations

This containerization setup provides a production-ready deployment while maintaining the local-first security model of Mind Map AI.

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
