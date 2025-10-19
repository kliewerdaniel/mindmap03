import pytest
from pathlib import Path
import tempfile
import shutil
from backend.app.db import db

@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    original_db_path = db.DB_PATH
    db.DB_PATH = Path(temp_dir) / "test.db"
    db.init_database()

    yield db.DB_PATH

    # Cleanup
    shutil.rmtree(temp_dir)
    db.DB_PATH = original_db_path

def test_insert_note(temp_db):
    """Test note insertion."""
    note_id = db.insert_note("test.md", "Test content", "/path/to/test.md")
    assert note_id > 0

    note = db.get_note(note_id)
    assert note['filename'] == "test.md"
    assert note['content'] == "Test content"
    assert note['processed'] == 0

def test_duplicate_note_hash(temp_db):
    """Test that duplicate content returns existing note_id."""
    note_id_1 = db.insert_note("test1.md", "Same content")
    note_id_2 = db.insert_note("test2.md", "Same content")

    assert note_id_1 == note_id_2

def test_insert_extract(temp_db):
    """Test extract insertion."""
    note_id = db.insert_note("test.md", "Test content")
    extract_json = {"nodes": [], "edges": []}
    extract_id = db.insert_extract(note_id, "llama3", extract_json, 0.95)

    assert extract_id > 0

    extracts = db.get_extracts_for_note(note_id)
    assert len(extracts) == 1
    assert extracts[0]['extractor_model'] == "llama3"

def test_mark_note_processed(temp_db):
    """Test marking note as processed."""
    note_id = db.insert_note("test.md", "Test content")
    db.mark_note_processed(note_id)

    note = db.get_note(note_id)
    assert note['processed'] == 1
