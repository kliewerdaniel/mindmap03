## Mind Map AI — Full Project Specification

**Project:** Mind Map AI — LLM-powered Personal Knowledge Graph (All Local)  
**Target:** Local-only stack (Next.js frontend, FastAPI backend, local LLM, SQLite, NetworkX graph).  
**Purpose:** Convert notes/journals/markdown into a browsable, queryable, and editable knowledge graph; provide semantic search and visualization; all inference and storage stays local.

---

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [User Stories & Flows](#2-user-stories--flows)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Technology Choices (Rationale)](#4-technology-choices-rationale)
5. [Data Models & Storage Design](#5-data-models--storage-design)
6. [LLM Strategy (Local Inference + Embeddings)](#6-llm-strategy)
7. [API Design (FastAPI)](#7-api-design)
8. [Frontend (Next.js)](#8-frontend)
9. [Graph Processing & Transformation Logic](#9-graph-processing--transformation-logic)
10. [Visualization Approach](#10-visualization-approach)
11. [File Structure & Example Files](#11-file-structure--example-files)
12. [Deployment / Local Dev Setup](#12-deployment--local-dev-setup)
13. [Testing & Validation Strategy](#13-testing--validation-strategy)
14. [Security & Privacy Considerations](#14-security--privacy-considerations)
15. [Performance & Scaling Notes](#15-performance--scaling-notes)
16. [Example Prompts & Extraction Templates](#16-example-prompts--extraction-templates)
17. [CLIne Handoff Notes](#17-cline-handoff-notes)
18. [Stretch Goals / Extensions](#18-stretch-goals--extensions)

---

## 1. Overview & Goals

**What it does:**
- Accepts local markdown/text notes (or pasted text)
- Uses a locally-hosted LLM to extract entities, concepts, relationships, and sentiment
- Stores raw notes in SQLite, embeddings in a local vector store, and graph relationships in a NetworkX graph persisted to disk
- Exposes an API for ingestion, querying, and editing
- Frontend (Next.js) provides an interactive visualization and editor for nodes/edges and a semantic search UI

**Constraints:**
- Everything local: inference, DB, vector store, UI served locally
- Offline-capable development workflow where possible
- Auditable transformations — every extraction stores source text and provenance

**Primary users:**
- You (the developer / blogger) building and experimenting; audience for blog: fellow vibe coders

---

## 2. User Stories & Flows

**User Stories:**
- As a user, I want to drop a folder of markdown into the app and have a graph generated automatically
- As a user, I want to click on a node and see the source passages and the LLM's extraction/provenance
- As a user, I want to semantically search my notes and get graph nodes as results
- As a user, I want to edit nodes/edges manually and commit changes
- As a user, I want exports: GraphML, GEXF, PNG snapshots

**Typical Flow:**
1. Drop or upload notes/folder or paste text
2. Backend reads files, extracts metadata, runs LLM extraction and embeddings
3. Save raw text to SQLite, embeddings to local vector store (Chroma or local Faiss), create/append nodes & edges to NetworkX graph
4. Frontend queries backend for graph and renders interactive visualization
5. User inspects nodes, opens provenance panel with source text and extracted labels
6. User edits a node/edge → backend updates NetworkX & SQLite
7. User exports or runs graph analytics (connected components, centrality)

---

## 3. High-Level Architecture

```
[ Next.js (frontend) ] <---> [ FastAPI (backend) ] <---> [Local LLM runtime (Ollama/Llama)]
                                   |-- SQLite (raw notes + metadata)
                                   |-- Vector DB (local Chroma / Faiss) (embeddings)
                                   |-- NetworkX (graph persisted as .gpickle / GraphML)
```

**Components:**
- **Frontend:** Next.js app (React). Interactive graph (react-cytoscapejs), note editor, search UI
- **Backend:** FastAPI for ingestion, graph management, search endpoints, admin endpoints
- **LLM runtime:** Ollama, Llama.cpp, or Dockerized local model backend (whichever you prefer). Used for extraction and for optional reasoning queries
- **Embeddings:** local sentence-transformer model (e.g., all-MiniLM or similar) or Ollama embedding endpoint (local)
- **Graph persistence:** NetworkX memory representation persisted to .gpickle / GraphML files, backed up in SQLite for quick metadata queries

---

## 4. Technology Choices (Rationale)

- **Next.js:** you're familiar with it; great for building modern UIs, server-side rendering for initial page load; can run entirely locally with `next dev` or `next start`
- **FastAPI:** lightweight, async, great for building REST APIs; easy to integrate with Python graph code and LLM libraries
- **NetworkX:** excellent for in-memory graph algorithms and flexible node/edge attributes; easy persistence to gpickle or GraphML
- **SQLite:** simple, file-based database for raw text and provenance; ACID, portable
- **Local LLM (Ollama / Llama):** keeps inference local. Ollama provides an easy local server experience; alternatives: llama.cpp or locally run Mistral/Gemma via supported runtimes
- **Embeddings:** local sentence-transformers or Ollama embeddings. Useful for fast semantic search
- **Vector DB:** lightweight local Chroma or Faiss if you want faster vector search than scanning SQLite
- **Visualization:** Cytoscape (via react-cytoscapejs) — good UX for graph exploration

---

## 5. Data Models & Storage Design

**SQLite Schema (Simplified):**

```sql
-- notes table: raw source markdown / text
CREATE TABLE notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT,
  content TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  source_path TEXT,     -- original path on disk if uploaded
  hash TEXT,            -- content hash for dedup
  processed BOOLEAN DEFAULT 0
);

-- extracts table: store entity extracts & provenance
CREATE TABLE extracts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  note_id INTEGER REFERENCES notes(id),
  extractor_model TEXT,
  extract_json TEXT,        -- store raw JSON output from LLM (entities, relationships)
  score REAL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- metadata table (optional)
CREATE TABLE metadata (
  key TEXT PRIMARY KEY,
  value TEXT
);
```

**NetworkX Graph Model:**
- **Node attributes:**
  - `id` (unique string; e.g., node:UUID or entity:<normalized_text>)
  - `label` (display name)
  - `type` (concept, person, place, idea, event, passage)
  - `provenance` (list of (note_id, span_start, span_end) tuples)
  - `embedding` (optional: vector; not stored directly in NetworkX but in vector DB with node id)
  - `created_at`, `updated_at`

- **Edge attributes:**
  - `type` (related_to, causes, elaborates, contradicts, similar_to, part_of)
  - `weight` (confidence score)
  - `extraction_id` (id in extracts table)
  - `provenance` (source spans)

**Persistence:**
- Save NetworkX to disk: `nx.write_gpickle(G, 'graph.gpickle')` or `nx.readwrite.gexf.write_gexf(G, path)` for export

---

## 6. LLM Strategy (Local Inference + Embeddings)

**Roles for LLM:**
1. **Extraction** — Given a text block, extract:
   - Entities (nouns, named entities)
   - Concepts (abstract ideas)
   - Relationships between entities/concepts with relation types and confidence
   - Short summaries for nodes or passages
   - Sentiment or metadata tags (mood, importance)

2. **Normalization** — Normalize entity names (e.g., "AI", "artificial intelligence" → canonical node)

3. **Reasoning / Querying** — Answer user questions by walking the graph and using the LLM to generate synthesis from node contents

4. **Rewrite / Summarize** — Generate node summaries for UI display

**Extraction Prompt Pattern:**
- Provide short instructions to extract JSON with a strict schema
- Include examples
- Ask model to return only JSON (machine-readable)

**Example Expected JSON:**

```json
{
  "nodes": [
    {"label": "sleep", "type": "concept", "span": [120, 170], "confidence": 0.95},
    {"label": "work", "type": "activity", "span": [0, 15], "confidence": 0.9}
  ],
  "edges": [
    {"source": "sleep", "target": "work", "type": "affects", "confidence": 0.87}
  ],
  "summary": "This passage mentions that sleep affects work energy..."
}
```

**Embeddings:**
- Use a local sentence-transformer model to embed each note and node label for semantic search
- Store vectors in local Chroma/Faiss, keyed by node id or note id

---

## 7. API Design (FastAPI)

**Core Endpoints:**

- `POST /api/ingest/file` — upload a file or zip of markdown files
- `POST /api/ingest/text` — post a text block for processing
- `GET /api/notes` — list notes
- `GET /api/notes/{id}` — get single note + extracts
- `POST /api/graph/build` — force rebuild graph from extracts
- `GET /api/graph` — get full graph or paginated
- `GET /api/graph/node/{id}` — get node details + provenance
- `POST /api/graph/node` — add/edit node
- `POST /api/graph/edge` — add/edit edge
- `POST /api/search/semantic` — body: `{"q": "...", "top_k": 10}`
- `GET /api/export/graph` — returns GraphML / GEXF / gpickle
- `POST /api/query/llm` — run a custom LLM prompt (local) — gated

**Example Ingestion Workflow:**
1. `POST /api/ingest/text` with `{"filename": "morning.md", "content": "I slept poorly..."}`
2. Backend saves to notes, returns note_id
3. Backend calls `extractor.process_note(note_id)` which:
   - runs LLM extraction
   - writes extracts row
   - updates NetworkX nodes & edges
   - indexes embeddings
4. Frontend polls `GET /api/notes/{id}` to check processed flag and show results

---

## 8. Frontend (Next.js)

**Pages:**
- `/` — Dashboard / quick summary and recent notes
- `/graph` — Full-screen interactive graph viewer
- `/note/[id]` — Note viewer + extraction provenance + edit controls
- `/search` — Semantic search interface
- `/settings` — LLM settings, model selection, embedding model, import/export

**Key Components:**
- `GraphCanvas` — react-cytoscapejs wrapper with pan/zoom, node click handlers
- `NodeDetailsPanel` — shows node metadata, provenance passages, edit buttons
- `NoteUploader` — drag & drop or folder selection
- `SemanticSearchBox` — search input with results mapped to nodes/notes
- `ModelControl` — choose local LLM / embeddings model, configure params

**UX Interactions:**
- Double-click node → open NodeDetailsPanel with source passages highlighted
- Right-click node → context menu: merge nodes, export node, delete node
- Lasso select → group operations
- Inline edit → on save, PATCH to `/api/graph/node`

---

## 9. Graph Processing & Transformation Logic

**Extraction Pipeline (per note):**
1. Read note content and optionally split into passages (by paragraphs or sliding window)
2. For each passage:
   - Send to LLM extraction prompt (strict JSON output)
   - Receive nodes & edges list, normalize labels
   - Assign node IDs based on normalization (e.g., slugify + checksum)
3. Merge nodes:
   - If normalized label already exists, merge provenance and update attributes (increment counts, update last_seen)
4. Create/Update edges:
   - Attach extraction_id and confidence
5. Store extracts and update `notes.processed = TRUE`
6. Index embeddings for note and nodes

**Normalization Heuristics:**
- Lowercase normalization + stopword stripping for short labels
- Use model to provide canonical name suggestion and disambiguation (LLM can propose canonical forms; store as canonical_label)
- Keep alias list on node attributes

**Conflict Resolution:**
- Keep original extraction raw store
- On conflicting edges (contradictory relations), create contradiction edge type or attach contradiction attribute with evidence list

---

## 10. Visualization Approach

**Recommendation:** Use react-cytoscapejs or cytoscape with cose or cola layout.

**Key Visual Cues:**
- Node color by type (concept, person, event)
- Node size by centrality (degree or eigenvector centrality)
- Edge thickness by weight (confidence)
- Hover tooltip shows top 1-2 provenance excerpts
- Click to open panel with full provenance + raw extract JSON + ability to edit

**Performance:**
- For large graphs, implement lazy loading and clustering. Only render subgraph around selected node by default (e.g., BFS to depth 2)
- Provide client-side search that requests filtered nodes from backend

---

## 11. File Structure & Example Files

```
mindmap-ai/
├─ backend/
│  ├─ app/
│  │  ├─ main.py                # FastAPI app
│  │  ├─ api/
│  │  │  ├─ ingest.py
│  │  │  ├─ graph.py
│  │  │  ├─ search.py
│  │  ├─ services/
│  │  │  ├─ extractor.py       # LLM extraction logic
│  │  │  ├─ embeddings.py
│  │  │  ├─ graph_store.py     # NetworkX wrapper + persistence
│  │  ├─ db/
│  │  │  ├─ schema.sql
│  │  │  ├─ db.py              # sqlite connection functions
│  ├─ requirements.txt
│  ├─ Dockerfile
├─ frontend/
│  ├─ package.json
│  ├─ next.config.js
│  ├─ src/
│  │  ├─ pages/
│  │  │  ├─ index.js
│  │  │  ├─ graph.js
│  │  │  ├─ note/[id].js
│  │  ├─ components/
│  │  │  ├─ GraphCanvas.jsx
│  │  │  ├─ NodePanel.jsx
│  │  │  ├─ SearchBox.jsx
│  ├─ Dockerfile
├─ models/                       # local LLM or pointers to models
├─ data/
│  ├─ notes/                     # sample markdown files
│  ├─ graph.gpickle
│  ├─ vectors/                    # vector DB files (Chroma/Faiss)
└─ README.md
```

---

## 12. Deployment / Local Dev Setup

**Development Steps (Summary):**
1. Install Python 3.10+ and Node 18+
2. **Backend:**
   - `cd backend`
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
   - Setup SQLite DB: run `app/db/schema.sql`
   - Configure local LLM endpoint in `app/config.py` (e.g., `http://localhost:11434` for Ollama)
   - `uvicorn app.main:app --reload --port 8000`
3. **Frontend:**
   - `cd frontend`
   - `npm install`
   - `npm run dev` (by default `http://localhost:3000`)
4. **LLM:**
   - Start Ollama or other local LLM runtime with the chosen model
5. Try `/api/ingest/text` via Postman or frontend uploader

**Docker (Optional):**
- Provide docker-compose with three services:
  - frontend (Next.js)
  - backend (FastAPI)
  - local LLM runtime (if using a docker-friendly image)
  - Volume mount `./data` and `./models`

---

## 13. Testing & Validation Strategy

**Unit Tests:**
- Test SQLite insert/read operations
- Test NetworkX persistence and loading
- Test `extractor.parse_output` function with sample JSON outputs (simulate LLM)

**Integration Tests:**
- Ingest sample markdown → run extraction → assert nodes count, edge count stable
- Semantic search correctness: query fixture questions and check expected node returns

**Manual QA:**
- Use a small set of notes with known relationships and ensure extraction and normalization produce expected outputs

---

## 14. Security & Privacy Considerations

- Everything local — no remote calls unless explicitly configured (e.g., to an optional cloud LLM). Default config should disable external network
- Raw notes stored in SQLite; consider encrypting the DB for extra privacy (e.g., using filesystem-level encryption or libs)
- LLM sandboxing: if using containerized LLM, ensure it's not exposed outside localhost
- Sanitize inputs to prevent injection-like threats into the backend shell or file system

---

## 15. Performance & Scaling Notes

- For many notes (thousands), NetworkX in-memory may become heavy. Strategies:
  - Shard graph by topic or file
  - Use persistent graph DB (Neo4j) as an upgrade path
  - Vector search: Faiss or Chroma with on-disk indexes recommended for large corpora
  - Batch extractions: process notes in parallel but throttle LLM calls to avoid resource exhaustion

---

## 16. Example Prompts & Extraction Templates

**Strict JSON Extractor Prompt (Short):**

```
System: You are a JSON extractor. Receive a short passage and return a JSON with nodes, edges, and summary. Return only valid JSON, nothing else. Use the schema below.

{
  "nodes": [{"label":..., "type":..., "span":[start,end], "confidence":float}],
  "edges": [{"source": "label_or_id", "target":"label_or_id", "type":"affects|relates_to|contradicts", "confidence":float}],
  "summary":"one-sentence summary"
}
```

**Example Instruction Body for Model:**

```
Passage:
"""
I haven't been sleeping well, which makes my work energy low and irritability higher. I want to improve exercise and sleep routine.
"""

Return JSON following schema: nodes: detect "sleep", "work energy", "irritability", "exercise", their types (concept/activity), edges such as sleep -> work energy (affects), include span character indexes and confidence scores between 0 and 1.
```

**Normalization Prompt (if using LLM to canonicalize):**
- Provide candidate aliases and ask model to choose canonical label and provide justification

---

## 17. CLIne Handoff Notes

**What to give CLIne later:**
- The full project README (this document)
- Preferred languages: Python (FastAPI), JS/TS (Next.js)
- Test data: a small `data/notes/` folder with 4–6 markdown files exhibiting overlapping concepts (to validate dedup and merging)
- Specify "All local" requirement and that LLM MUST be local; provide model preference (e.g., llama-3 via Ollama)
- Ask for:
  - Implementation of the API endpoints described
  - Basic Next.js frontend with GraphCanvas & NodePanel
  - A minimal extraction prompt (as provided) and an extractor harness that can be swapped for different LLM endpoints easily
  - Provide acceptance tests:
    - Ingest sample notes and produce at least N nodes and M edges (numbers based on sample)
    - Export GraphML and confirm at least one node with provenance exists

---

## 18. Stretch Goals / Extensions

- Graph analytics dashboard: centrality, communities (Louvain), timeline of nodes by created_at
- Versioning & diffs: maintain history of node edits and allow rollback
- Local fine-tuning: fine-tune an LLM locally on your own notes for improved extraction
- Sync to Obsidian or local vault: keep files in sync
- Biometric integration: pair node tags with daily metrics (sleep HR from device) — for the journaling use case
- Export to Neo4j for larger-scale graph storage or use as a migration path

---

## Appendix — Sample Code Snippets

**FastAPI Ingestion Skeleton (Illustrative):**

```python
# backend/app/api/ingest.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.extractor import process_text
from ..db.db import insert_note

router = APIRouter()

class IngestRequest(BaseModel):
    filename: str
    content: str

@router.post("/text")
async def ingest_text(payload: IngestRequest):
    note_id = insert_note(payload.filename, payload.content)
    # process in background or synchronous depending on config:
    result = process_text(note_id, payload.content)  # calls LLM
    return {"note_id": note_id, "result": result}
```

**NetworkX Persistence Example:**

```python
import networkx as nx
G = nx.Graph()
G.add_node("sleep", type="concept", label="sleep")
G.add_node("work", type="activity", label="work energy")
G.add_edge("sleep", "work", type="affects", weight=0.95)
nx.write_gpickle(G, "data/graph.gpickle")
# load:
G2 = nx.read_gpickle("data/graph.gpickle")
```

**Example LLM Call (Pseudo):**

```python
def call_local_llm(prompt: str) -> dict:
    # Example using requests to an Ollama-like local endpoint
    import requests
    r = requests.post("http://localhost:11434/api/text", json={"prompt": prompt})
    return r.json()
```

---

## Final Notes 

- This project is perfect for vibe-coding: incremental wins (drop a note → see a node), clear visuals (graph grows as you feed it), and deep future-proofing (persisted graph + raw extracts)
- Keep everything auditable — that'll make your blog narrative strong: "I fed my journal to a local LLM and watched my mind's topology appear"
- Start small: one FastAPI endpoint + one Next.js page with a small sample Markdown folder. Graduate to background processing and better UI after you confirm extraction quality

---