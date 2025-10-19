import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Any
import hashlib
import json
from ..config import settings

def get_connection() -> sqlite3.Connection:
    """Get SQLite connection with row factory."""
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize database with schema."""
    # Ensure parent directory exists
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path) as f:
        schema = f.read()

    conn = get_connection()
    conn.executescript(schema)
    conn.commit()
    conn.close()

def insert_note(filename: str, content: str, source_path: Optional[str] = None) -> int:
    """Insert note and return note_id. Skip if hash exists."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    conn = get_connection()
    cursor = conn.cursor()

    # Check if note with same hash exists
    cursor.execute("SELECT id FROM notes WHERE hash = ?", (content_hash,))
    existing = cursor.fetchone()

    if existing:
        conn.close()
        return existing[0]

    cursor.execute(
        "INSERT INTO notes (filename, content, source_path, hash) VALUES (?, ?, ?, ?)",
        (filename, content, source_path, content_hash)
    )
    note_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return note_id

def insert_extract(note_id: int, extractor_model: str, extract_json: Dict, score: Optional[float] = None) -> int:
    """Insert extraction result."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO extracts (note_id, extractor_model, extract_json, score) VALUES (?, ?, ?, ?)",
        (note_id, extractor_model, json.dumps(extract_json), score)
    )
    extract_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return extract_id

def mark_note_processed(note_id: int):
    """Mark note as processed after extraction."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE notes SET processed = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()

def get_note(note_id: int) -> Optional[Dict]:
    """Retrieve note by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None

def get_all_notes() -> List[Dict]:
    """Retrieve all notes."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notes ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]

def get_extracts_for_note(note_id: int) -> List[Dict]:
    """Retrieve all extracts for a given note."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM extracts WHERE note_id = ? ORDER BY created_at DESC", (note_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]
