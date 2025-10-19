from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from pathlib import Path
from ..config import settings
import numpy as np

class EmbeddingStore:
    """Manages embeddings using sentence-transformers and ChromaDB."""

    def __init__(self):
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(settings.vector_db_path)
        ))

        # Get or create collections
        self.notes_collection = self.chroma_client.get_or_create_collection(
            name="notes",
            metadata={"description": "Note embeddings"}
        )

        self.nodes_collection = self.chroma_client.get_or_create_collection(
            name="nodes",
            metadata={"description": "Node label embeddings"}
        )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def index_note(self, note_id: int, content: str, metadata: Dict = None):
        """Index a note for semantic search."""
        embedding = self.embed_text(content)

        self.notes_collection.add(
            ids=[f"note:{note_id}"],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}]
        )

    def index_node(self, node_id: str, label: str, node_type: str, metadata: Dict = None):
        """Index a node for semantic search."""
        embedding = self.embed_text(label)

        self.nodes_collection.add(
            ids=[node_id],
            embeddings=[embedding],
            documents=[label],
            metadatas=metadata or {}
        )

    def search_notes(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search notes by semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of results with note_id, content, and similarity score
        """
        query_embedding = self.embed_text(query)

        results = self.notes_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        output = []
        for i, note_ref in enumerate(results['ids'][0]):
            note_id = int(note_ref.split(':')[1])
            output.append({
                'note_id': note_id,
                'content': results['documents'][0][i],
                'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
            })

        return output

    def search_nodes(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search nodes by semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of results with node_id, label, and similarity score
        """
        query_embedding = self.embed_text(query)

        results = self.nodes_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        output = []
        for i, node_id in enumerate(results['ids'][0]):
            output.append({
                'node_id': node_id,
                'label': results['documents'][0][i],
                'score': 1 - results['distances'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
            })

        return output

    def delete_note(self, note_id: int):
        """Remove note from index."""
        try:
            self.notes_collection.delete(ids=[f"note:{note_id}"])
        except:
            pass  # Note may not exist in index

    def delete_node(self, node_id: str):
        """Remove node from index."""
        try:
            self.nodes_collection.delete(ids=[node_id])
        except:
            pass  # Node may not exist in index

# Global instance
_embedding_store = None

def get_embedding_store() -> EmbeddingStore:
    """Get or create global embedding store instance."""
    global _embedding_store
    if _embedding_store is None:
        _embedding_store = EmbeddingStore()
    return _embedding_store

def init_embeddings():
    """Initialize embedding store on startup."""
    global _embedding_store
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    _embedding_store = EmbeddingStore()
