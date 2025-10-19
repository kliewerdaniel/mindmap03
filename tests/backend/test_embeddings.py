import pytest
from backend.app.services.embeddings import EmbeddingStore
import tempfile
from pathlib import Path
import shutil

@pytest.fixture
def temp_embedding_store():
    """Create temporary embedding store."""
    temp_dir = Path(tempfile.mkdtemp())

    # Mock settings
    from backend.app import config
    original_path = config.settings.vector_db_path
    config.settings.vector_db_path = temp_dir

    store = EmbeddingStore()

    yield store

    # Cleanup
    shutil.rmtree(temp_dir)
    config.settings.vector_db_path = original_path

def test_embed_text(temp_embedding_store):
    """Test text embedding generation."""
    embedding = temp_embedding_store.embed_text("test content")

    assert isinstance(embedding, list)
    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
    assert all(isinstance(x, float) for x in embedding)

def test_index_and_search_notes(temp_embedding_store):
    """Test note indexing and search."""
    # Index notes
    temp_embedding_store.index_note(1, "Machine learning is transforming AI")
    temp_embedding_store.index_note(2, "I love cooking pasta with fresh tomatoes")
    temp_embedding_store.index_note(3, "Neural networks and deep learning")

    # Search
    results = temp_embedding_store.search_notes("artificial intelligence", top_k=2)

    assert len(results) <= 2
    assert results[0]['note_id'] in [1, 3]  # Should match AI-related notes
    assert 'score' in results[0]

def test_index_and_search_nodes(temp_embedding_store):
    """Test node indexing and search."""
    # Index nodes
    temp_embedding_store.index_node("node:1", "machine learning", "concept")
    temp_embedding_store.index_node("node:2", "pasta", "concept")
    temp_embedding_store.index_node("node:3", "deep learning", "concept")

    # Search
    results = temp_embedding_store.search_nodes("AI algorithms", top_k=2)

    assert len(results) <= 2
    # Should prioritize ML-related nodes
    top_result_label = results[0]['label'].lower()
    assert any(term in top_result_label for term in ['machine', 'learning', 'deep'])

def test_delete_note(temp_embedding_store):
    """Test note deletion from index."""
    temp_embedding_store.index_note(1, "test content")

    # Verify indexed
    results = temp_embedding_store.search_notes("test", top_k=5)
    assert any(r['note_id'] == 1 for r in results)

    # Delete
    temp_embedding_store.delete_note(1)

    # Verify removed
    results = temp_embedding_store.search_notes("test", top_k=5)
    assert not any(r['note_id'] == 1 for r in results)
