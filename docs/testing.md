## Frontend Testing

### Component Testing (React Testing Library)

Test coverage for:
- GraphCanvas render with sample data
- NodeDetailsPanel data display
- User interactions (click, double-click)
- Loading and error states

### E2E Testing (Playwright - Future Phase)

Critical user flows:
1. Load graph page → View graph → Click node → View details
2. Search for node → Select from results → Navigate to graph
3. Upload note → Wait for processing → Verify graph updated

## Integration Testing

### Setup

1. Ensure backend is running:
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

2. Ensure local LLM is running (Ollama):
```bash
ollama serve
```

3. Run integration tests:
```bash
pytest tests/integration/test_full_pipeline.py -v
```

### Acceptance Tests

#### Test 1: Sample Notes Ingestion

**Objective**: Verify complete pipeline from ingestion to graph creation.

**Steps**:
1. Ingest all files from `data/notes/`
2. Wait for processing completion
3. Verify graph contains nodes (N > 0) and edges (M ≥ 0)

**Success Criteria**:
- All notes marked as processed
- Graph contains extracted nodes
- Each node has provenance data

#### Test 2: Provenance Verification

**Objective**: Ensure exported graph contains full provenance.

**Steps**:
1. Export graph as GraphML
2. Parse and verify structure
3. Check for provenance attributes on nodes

**Success Criteria**:
- Export completes successfully
- GraphML contains valid XML
- At least one node has provenance attribute with source reference

### Manual Testing Checklist

- [ ] Upload single markdown file via frontend
- [ ] Verify note appears in database: `sqlite3 data/mindmap.db "SELECT * FROM notes;"`
- [ ] Verify extraction in database: `sqlite3 data/mindmap.db "SELECT * FROM extracts;"`
- [ ] Navigate to `/graph` page and verify visualization
- [ ] Click node and verify details panel opens
- [ ] Perform semantic search and verify results
- [ ] Export graph and verify file downloads
