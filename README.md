# Mind Map AI - Local LLM-powered Personal Knowledge Graph

A comprehensive personal knowledge management system that builds interactive mind maps from your notes using local Large Language Models (LLMs) and vector embeddings.

## Overview

Mind Map AI transforms your personal notes, documents, and research into an intelligent, navigable knowledge graph. The system uses local LLMs for entity and relationship extraction, creating semantic connections between your ideas without relying on external APIs or cloud services.

## Key Features

### 🧠 Intelligent Knowledge Extraction
- **Local LLM Integration**: Uses Ollama or Llama.cpp for private, local processing
- **Entity Recognition**: Automatically identifies people, places, concepts, and organizations
- **Relationship Mapping**: Discovers connections and associations between entities
- **Confidence Scoring**: Provides reliability metrics for extracted information

### 🗂️ Smart Organization
- **NetworkX Graph Engine**: In-memory graph processing with disk persistence
- **Semantic Search**: Vector embeddings for similarity-based content discovery
- **SQLite Backend**: Efficient metadata and provenance tracking
- **Version Control**: Graph state management and rollback capabilities

### 🎨 Interactive Visualization
- **Next.js Frontend**: Modern, responsive web interface
- **Cytoscape.js Integration**: Dynamic, interactive graph visualization
- **Real-time Updates**: Live graph modifications and filtering
- **Export Capabilities**: Multiple format support (GraphML, JSON, PNG, PDF)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js       │    │   FastAPI        │    │   Local LLM     │
│   Frontend      │◄──►│   Backend        │◄──►│   (Ollama)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Cytoscape.js   │    │   SQLite         │    │   Sentence-     │
│  Visualization  │    │   Database       │    │   Transformers  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  React Flow     │    │   NetworkX       │    │   Chroma/Faiss  │
│  Components     │    │   Graph Engine   │    │   Vector Store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **NetworkX**: Graph creation and analysis
- **SQLite**: Lightweight relational database
- **SQLAlchemy**: Database ORM and migrations
- **Ollama/Llama.cpp**: Local LLM integration
- **Sentence-Transformers**: Text embeddings
- **Chroma/Faiss**: Vector database for semantic search

### Frontend
- **Next.js 14**: React framework with App Router
- **React Cytoscape.js**: Interactive graph visualization
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first styling
- **React Query**: Server state management

### Data Storage
- **Graph Data**: NetworkX with .gpickle/GraphML persistence
- **Vector Embeddings**: Chroma DB or FAISS indexing
- **Metadata**: SQLite with provenance tracking
- **File Storage**: Local filesystem with organization

## Installation

### Prerequisites
- Python 3.8+
- Node.js 18+
- Ollama (for local LLM) or Llama.cpp
- Git

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd mindmap-ai
```

2. Backend setup:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

3. Frontend setup:
```bash
cd frontend
npm install
# or
yarn install
```

4. Start services:
```bash
# Backend (from backend directory)
uvicorn app.main:app --reload

# Frontend (from frontend directory)
npm run dev
```

## Usage

1. **Upload Notes**: Import your documents, notes, or research files
2. **Extract Knowledge**: The system processes content through the local LLM
3. **Explore Connections**: Navigate the interactive knowledge graph
4. **Semantic Search**: Find related concepts using natural language queries
5. **Export & Share**: Save graphs in multiple formats

## Configuration

The system supports extensive configuration through environment variables and config files:

- **LLM Settings**: Model selection, temperature, max tokens
- **Graph Parameters**: Node/edge thresholds, clustering options
- **Storage Paths**: Custom directories for data persistence
- **UI Preferences**: Visualization themes, layout algorithms

## Privacy & Security

- **Local-Only**: All processing happens on your machine
- **No External APIs**: Complete data sovereignty
- **Encryption**: Optional encryption for stored graphs
- **Access Control**: User authentication and authorization

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/guides/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Multi-modal content support (PDF, images, audio)
- [ ] Advanced graph algorithms (community detection, centrality)
- [ ] Collaborative features (when privacy allows)
- [ ] Mobile applications
- [ ] Plugin system for custom extractors

## Support

For support and questions:
- [Documentation](docs/)
- [Issues](https://github.com/yourusername/mindmap-ai/issues)
- [Discussions](https://github.com/yourusername/mindmap-ai/discussions)

---

*Built with ❤️ for personal knowledge management*
