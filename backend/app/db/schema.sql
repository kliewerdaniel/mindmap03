-- Table: notes
-- Stores raw markdown/text content with metadata
CREATE TABLE notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  source_path TEXT,
  hash TEXT UNIQUE,  -- Content hash for deduplication
  processed BOOLEAN DEFAULT 0  -- Flag for extraction completion
);

-- Table: extracts
-- Stores LLM extraction results with provenance
CREATE TABLE extracts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  extractor_model TEXT NOT NULL,  -- Model identifier (e.g., "llama3-8b")
  extract_json TEXT NOT NULL,     -- Raw JSON output from LLM
  score REAL,                      -- Confidence/quality score
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (note_id) REFERENCES notes(id)
);

-- Table: metadata
-- Key-value store for system metadata
CREATE TABLE metadata (
  key TEXT PRIMARY KEY,
  value TEXT,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_notes_hash ON notes(hash);
CREATE INDEX idx_notes_processed ON notes(processed);
CREATE INDEX idx_extracts_note_id ON extracts(note_id);
