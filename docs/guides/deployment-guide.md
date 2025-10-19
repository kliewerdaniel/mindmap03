# Deployment Guide - Mind Map AI

This guide covers various deployment strategies for Mind Map AI, from local development to production environments.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)

## Local Development

### Prerequisites

- Python 3.8+
- Node.js 18+
- Git
- Ollama (recommended) or Llama.cpp

### Quick Start

1. **Clone the repository:**
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

3. **Frontend setup:**
```bash
cd ../frontend
npm install
```

4. **Start services:**
```bash
# Terminal 1 - Backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

5. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Docker Deployment

### Docker Compose (Recommended)

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./backend/logs:/app/logs
    environment:
      - PYTHONPATH=/app
    depends_on:
      - ollama

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

### Backend Dockerfile

```dockerfile
# backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Create necessary directories
RUN mkdir -p /app/data/graphs /app/data/embeddings /app/logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production image
FROM node:18-alpine AS runner

WORKDIR /app

RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

USER nextjs

EXPOSE 3000

CMD ["npm", "start"]
```

### Deployment Commands

```bash
# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down

# Update and redeploy
docker-compose pull
docker-compose up --build -d
```

## Production Deployment

### System Requirements

**Minimum Specifications:**
- CPU: 4+ cores (2.4 GHz+)
- RAM: 8GB minimum, 16GB recommended
- Storage: 50GB SSD
- Network: 100 Mbps

**Recommended Specifications:**
- CPU: 8+ cores (3.0 GHz+)
- RAM: 32GB
- Storage: 100GB NVMe SSD
- Network: 1 Gbps

### Environment Configuration

```bash
# backend/.env.production
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v1

# Database
DATABASE_URL=sqlite:///./data/metadata.db

# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama2:13b
LLM_BASE_URL=http://ollama:11434

# Storage Paths
GRAPH_STORAGE_PATH=./data/graphs
EMBEDDING_STORAGE_PATH=./data/embeddings

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Performance
MAX_WORKERS=4
CHUNK_SIZE=1000
```

### Reverse Proxy Setup (Nginx)

```nginx
# nginx.conf
upstream mindmap_backend {
    server backend:8000;
}

upstream mindmap_frontend {
    server frontend:3000;
}

server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        proxy_pass http://mindmap_frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Backend API
    location /api/ {
        proxy_pass http://mindmap_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for real-time updates
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static files caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        proxy_pass http://mindmap_frontend;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### SSL/TLS Configuration

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Generate SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (add to crontab)
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Cloud Deployment

### AWS Deployment

#### EC2 Deployment

1. **Launch EC2 instance:**
```bash
# Ubuntu 22.04 LTS, t3.large or larger
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.large \
    --key-name your-key-pair \
    --security-groups mindmap-sg
```

2. **Install Docker:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
```

3. **Deploy application:**
```bash
git clone <repository-url>
cd mindmap-ai
docker-compose up -d
```

#### ECS Fargate Deployment

```yaml
# ecs-task-definition.json
{
    "family": "mindmap-ai",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "2048",
    "memory": "8192",
    "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::account:role/mindmapTaskRole",
    "containerDefinitions": [
        {
            "name": "backend",
            "image": "your-registry/backend:latest",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {"name": "DATABASE_URL", "value": "postgresql://..."}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/mindmap-ai",
                    "awslogs-region": "us-east-1"
                }
            }
        }
    ]
}
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push container
gcloud builds submit --tag gcr.io/your-project/mindmap-ai

# Deploy to Cloud Run
gcloud run deploy mindmap-ai \
    --image gcr.io/your-project/mindmap-ai \
    --platform managed \
    --region us-central1 \
    --memory 8Gi \
    --cpu 4 \
    --port 8000 \
    --allow-unauthenticated
```

#### GKE Deployment

```yaml
# kubernetes-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mindmap-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mindmap-ai
  template:
    metadata:
      labels:
        app: mindmap-ai
    spec:
      containers:
      - name: backend
        image: gcr.io/your-project/backend:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
---
apiVersion: v1
kind: Service
metadata:
  name: mindmap-ai-service
spec:
  selector:
    app: mindmap-ai
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Azure Deployment

#### Container Instances

```bash
# Create resource group
az group create --name mindmap-ai-rg --location eastus

# Deploy container
az container create \
    --resource-group mindmap-ai-rg \
    --name mindmap-ai \
    --image your-registry/mindmap-ai:latest \
    --ports 8000 \
    --cpu 4 \
    --memory 8 \
    --environment-variables API_HOST=0.0.0.0
```

## Monitoring and Maintenance

### Health Checks

#### Backend Health Check

```python
# app/api/v1/endpoints/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """Application health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@router.get("/readiness")
async def readiness_check():
    """Kubernetes readiness probe."""
    # Check database connectivity, LLM availability, etc.
    return {"status": "ready"}
```

#### Docker Health Checks

```yaml
# docker-compose.yml (updated)
services:
  backend:
    # ... existing config ...
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Logging Configuration

#### Production Logging

```python
# app/core/logging.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger("mindmap_ai")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
```

### Backup Strategy

#### Automated Backups

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
cp data/metadata.db $BACKUP_DIR/metadata_$TIMESTAMP.db

# Backup graphs
tar -czf $BACKUP_DIR/graphs_$TIMESTAMP.tar.gz -C data graphs/

# Backup embeddings
tar -czf $BACKUP_DIR/embeddings_$TIMESTAMP.tar.gz -C data embeddings/

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup completed: $TIMESTAMP"
```

#### Cron Configuration

```bash
# Add to crontab for daily backups at 2 AM
crontab -e
# Add: 0 2 * * * /path/to/mindmap-ai/scripts/backup.sh
```

### Performance Monitoring

#### Metrics Collection

```python
# app/core/monitoring.py
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

def monitor_requests():
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(method='POST', endpoint='/upload').inc()
                return result
            finally:
                REQUEST_DURATION.observe(time.time() - start_time)
        return wrapper
    return decorator
```

#### Grafana Dashboard

Key metrics to monitor:
- Request rate and latency
- Error rates by endpoint
- Graph size and growth
- LLM processing times
- Memory and CPU usage
- Disk space utilization

### Security Hardening

#### Firewall Configuration

```bash
# UFW (Uncomplicated Firewall)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

#### Security Updates

```bash
#!/bin/bash
# scripts/security-updates.sh

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Update Python packages
pip install --upgrade pip
pip install -r requirements.txt --upgrade

# Update Node.js packages
npm audit fix
npm update

# Security scan
safety check
npm audit

echo "Security updates completed"
```

### Troubleshooting

#### Common Issues

**High Memory Usage:**
```bash
# Check memory usage
docker stats

# Clear Docker cache if needed
docker system prune -a
```

**Slow LLM Processing:**
```bash
# Check Ollama status
curl http://localhost:11434/api/version

# Monitor system resources
htop
```

**Database Connection Issues:**
```bash
# Check database file permissions
ls -la data/metadata.db

# Test database connectivity
sqlite3 data/metadata.db "SELECT name FROM sqlite_master LIMIT 1;"
```

#### Log Analysis

```bash
# View recent logs
tail -f backend/logs/app.log

# Search for errors
grep -i error backend/logs/app.log

# Filter by date
journalctl --since "2024-01-01" --until "2024-01-31"
```

## Scaling Considerations

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# nginx load balancer
upstream mindmap_backend {
    least_conn;
    server backend1:8000 weight=3;
    server backend2:8000 weight=3;
    server backend3:8000 weight=1;
}

server {
    listen 80;
    location / {
        proxy_pass http://mindmap_backend;
        # ... proxy settings ...
    }
}
```

### Database Scaling

#### Read Replicas

For high-read scenarios, consider:
- SQLite read replicas for complex queries
- Connection pooling for better resource utilization
- Query optimization and caching

#### Migration Strategy

```sql
-- Example migration script
ALTER TABLE entities ADD COLUMN last_accessed DATETIME;
CREATE INDEX idx_entities_last_accessed ON entities(last_accessed);
```

## Compliance and Regulations

### GDPR Compliance

- **Data Minimization**: Only collect necessary information
- **Right to Erasure**: Implement data deletion capabilities
- **Data Portability**: Export functionality for user data
- **Consent Management**: Clear privacy policy and consent collection

### Data Retention

```python
# app/core/retention.py
class DataRetention:
    def __init__(self, retention_days: int = 2555):  # 7 years
        self.retention_days = retention_days

    def cleanup_old_data(self):
        """Remove data older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        # Clean old graphs
        # Clean old embeddings
        # Clean old metadata
        pass
```

## Support and Maintenance

### Regular Maintenance Tasks

1. **Daily**: Monitor logs and system health
2. **Weekly**: Review and optimize slow queries
3. **Monthly**: Security updates and dependency updates
4. **Quarterly**: Performance review and capacity planning

### Support Channels

- **Documentation**: docs/ directory
- **Issue Tracking**: GitHub Issues
- **Community**: GitHub Discussions
- **Email Support**: support@your-domain.com

### Update Procedures

```bash
# Graceful update process
1. Notify users of upcoming maintenance
2. Create database backup
3. Deploy new version to staging
4. Test thoroughly in staging
5. Deploy to production during low-traffic period
6. Monitor for issues post-deployment
7. Rollback if necessary
```

This deployment guide provides comprehensive coverage for deploying Mind Map AI in various environments. Always test thoroughly in a staging environment before production deployment.
