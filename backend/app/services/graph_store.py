import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
from datetime import datetime
from ..config import settings

class GraphStore:
    """Manages NetworkX graph with disk persistence."""

    def __init__(self, graph_path: Optional[Path] = None):
        self.graph_path = graph_path or settings.graph_path
        self.graph = self._load_graph()

    def _load_graph(self) -> nx.Graph:
        """Load graph from disk or create new."""
        if self.graph_path.exists():
            try:
                with open(self.graph_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading graph: {e}. Creating new graph.")
                return nx.Graph()
        else:
            return nx.Graph()

    def save(self):
        """Persist graph to disk."""
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)

    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        provenance: List[Tuple[int, int, int]] = None,
        **kwargs
    ) -> str:
        """
        Add or update node in graph.

        Args:
            node_id: Unique node identifier
            label: Display name
            node_type: Type (concept, person, place, idea, event, passage)
            provenance: List of (note_id, span_start, span_end) tuples
            **kwargs: Additional attributes (embedding, metadata, etc.)

        Returns:
            node_id
        """
        if self.graph.has_node(node_id):
            # Update existing node
            existing = self.graph.nodes[node_id]
            existing['label'] = label
            existing['type'] = node_type

            # Merge provenance
            existing_prov = existing.get('provenance', [])
            new_prov = provenance or []
            existing['provenance'] = existing_prov + [p for p in new_prov if p not in existing_prov]

            existing['updated_at'] = datetime.now().isoformat()
            existing.update(kwargs)
        else:
            # Add new node
            self.graph.add_node(
                node_id,
                label=label,
                type=node_type,
                provenance=provenance or [],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                **kwargs
            )

        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: float = 1.0,
        extraction_id: Optional[int] = None,
        provenance: Optional[List[Tuple[int, int, int]]] = None,
        **kwargs
    ):
        """
        Add or update edge in graph.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Relationship type (related_to, causes, elaborates, etc.)
            weight: Confidence score (0-1)
            extraction_id: Reference to extracts table
            provenance: Source spans
            **kwargs: Additional attributes
        """
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            raise ValueError(f"Both nodes must exist before adding edge: {source} -> {target}")

        if self.graph.has_edge(source, target):
            # Update existing edge
            existing = self.graph.edges[source, target]
            existing['type'] = edge_type
            existing['weight'] = weight
            existing['extraction_id'] = extraction_id
            existing['provenance'] = provenance or []
            existing['updated_at'] = datetime.now().isoformat()
            existing.update(kwargs)
        else:
            # Add new edge
            self.graph.add_edge(
                source,
                target,
                type=edge_type,
                weight=weight,
                extraction_id=extraction_id,
                provenance=provenance or [],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                **kwargs
            )

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node attributes."""
        if self.graph.has_node(node_id):
            data = dict(self.graph.nodes[node_id])
            data['id'] = node_id
            return data
        return None

    def get_all_nodes(self) -> List[Dict]:
        """Get all nodes with attributes."""
        return [
            {'id': node_id, **dict(attrs)}
            for node_id, attrs in self.graph.nodes(data=True)
        ]

    def get_edges(self, node_id: Optional[str] = None) -> List[Dict]:
        """Get edges, optionally filtered by node."""
        if node_id:
            edges = self.graph.edges(node_id, data=True)
        else:
            edges = self.graph.edges(data=True)

        return [
            {'source': u, 'target': v, **attrs}
            for u, v, attrs in edges
        ]

    def delete_node(self, node_id: str):
        """Remove node and associated edges."""
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)

    def delete_edge(self, source: str, target: str):
        """Remove edge."""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[str]:
        """Get neighboring nodes up to specified depth."""
        if not self.graph.has_node(node_id):
            return []

        neighbors = set()
        current_level = {node_id}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.neighbors(node))
            neighbors.update(next_level)
            current_level = next_level

        return list(neighbors)

    def get_subgraph(self, node_id: str, depth: int = 2) -> Dict:
        """Get subgraph around node for visualization."""
        neighbors = self.get_neighbors(node_id, depth)
        nodes_to_include = [node_id] + neighbors

        subgraph = self.graph.subgraph(nodes_to_include)

        return {
            'nodes': [
                {'id': n, **dict(attrs)}
                for n, attrs in subgraph.nodes(data=True)
            ],
            'edges': [
                {'source': u, 'target': v, **attrs}
                for u, v, attrs in subgraph.edges(data=True)
            ]
        }

    def compute_centrality(self, metric: str = 'degree') -> Dict[str, float]:
        """Compute centrality metrics for visualization."""
        if metric == 'degree':
            return nx.degree_centrality(self.graph)
        elif metric == 'eigenvector':
            try:
                return nx.eigenvector_centrality(self.graph, max_iter=1000)
            except:
                return nx.degree_centrality(self.graph)  # Fallback
        elif metric == 'betweenness':
            return nx.betweenness_centrality(self.graph)
        else:
            return nx.degree_centrality(self.graph)

    def export_graphml(self, output_path: Path):
        """Export graph to GraphML format."""
        nx.write_graphml(self.graph, str(output_path))

    def export_gexf(self, output_path: Path):
        """Export graph to GEXF format."""
        nx.write_gexf(self.graph, str(output_path))

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
        }


# Global instance
_graph_store = None

def get_graph_store() -> GraphStore:
    """Get or create global graph store instance."""
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
    return _graph_store

def init_graph():
    """Initialize graph store on startup."""
    global _graph_store
    _graph_store = GraphStore()
